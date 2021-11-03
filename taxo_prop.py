import argparse
import os

from scipy import sparse

from data_loader.dataset import TaxonDataset

import sys
from model.model import BatchModel
import pickle
import torch
import dgl
from preprocess.taxo_dgl_construction import update_ancestor, construct_hetergraph, sparse_mx_to_torch_sparse_tensor
from time import time
import numpy as np
import torch.nn as nn
from collections import defaultdict
import torch.nn.functional as F
from IPython import embed
#from sklearn.metrics import ndcg_score

from sklearn.metrics import (auc, f1_score, precision_recall_curve,
                             roc_auc_score)
from torch.nn.parameter import Parameter
from ogb.linkproppred import Evaluator

from annoy import AnnoyIndex
from tqdm import tqdm


dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(dir_path, os.pardir))

def evaluate(pred, label):
    pred = np.array(pred).reshape(-1,)
    label = np.array(label, dtype=bool).reshape(-1,)
    sorted_pred = sorted(pred, reverse=True)
    sorted_label = sorted(label, reverse=True)
    true_num = np.sum(label)
    threshold = sorted_pred[true_num]

    y_pred = np.zeros(len(pred), dtype=bool)

    #y_pred[pred>=0.5] = True
    y_pred[pred>=threshold] = True
    # embed()
    ps, rs, _ = precision_recall_curve(label, pred)
    pred_result = y_pred == label
    return roc_auc_score(label, pred), f1_score(label, y_pred), auc(rs, ps)

def generate_pos_taxon_edges(src_data_path, fos2idx):
    # src_data_path = '/shared/data2/qiz3/taxograph/dataset/MAG'
    # src_data_path = '/shared/data2/qiz3/taxograph/dataset/MeSH'
    miss_cnt = 0
    with open(src_data_path + '/taxonomy.txt') as IN, open(src_data_path+'/taxonomy_network.txt', 'w') as OUT:
        pos_edges = set()    
        for line in IN:
            data = line.strip().split()
            p = int(data[0]) if 'MAG' in src_data_path else data[0]
            if p not in fos2idx:
                miss_cnt += 1
                continue
            p_id = fos2idx[p]
            for c in data[1:]:
                c = int(c) if 'MAG' in src_data_path else c
                if c not in fos2idx:
                    miss_cnt += 1
                    continue
                c_id = fos2idx[c]
                pos_edges.add((p_id, c_id))
                OUT.write(f'{p_id} {c_id} 1\n')
    #taxon_data = TaxonDataset('computer_science', path=parent_path+'/MAG-CS')
    #embed()
    return list(pos_edges)


def read_taxon_features(data_src_folder):
    from scipy import sparse
    with open(data_src_folder+'/paper_fos_score_vec_dict.pkl', 'rb') as handle:
        paper_fos_score_vec_dict = pickle.load(handle)
    with open(data_src_folder+'/idx2name_dict.pkl', 'rb') as handle:
        idx2name_dict = pickle.load(handle)

    name2fos_dict = {}
    # This part is problematic, fix it
    fos_name_path = data_src_folder+'/id2label.txt'
    with open(fos_name_path, 'r') as f:
        for line in f:
            id_str, fos_name = line.strip().split('\t')
            name2fos_dict[fos_name] = int(id_str) if 'MAG' in data_src_folder else id_str

    fos2idx = {}
    for i in idx2name_dict:
        if idx2name_dict[i] in name2fos_dict:
            fos2idx[name2fos_dict[idx2name_dict[i]]] = i

    pos_edges = generate_pos_taxon_edges(data_src_folder, fos2idx)

    sp_tensor_fos_feature, init_taxo_emb = None, None
    if 'MAG' in data_src_folder:
        vec_cache = []
        for p_idx in range(len(p_idx2id_dict)):
            vec_cache.append(paper_fos_score_vec_dict[p_idx2id_dict[p_idx]])
        sp_fos_feature = sparse.vstack(vec_cache)
        sp_tensor_fos_feature = sparse_mx_to_torch_sparse_tensor(sp_fos_feature)

        if True:
            with open('taxonomy_emb.txt', 'r') as IN:
                num_x, num_dim = IN.readline().strip().split(' ')
                init_taxo_emb = torch.zeros((sp_tensor_fos_feature.shape[1]+1, int(num_dim)))
                IN.readline()
                for line in IN:
                    tmp = line.strip().split(' ')
                    init_taxo_emb[int(tmp[0]), :] = torch.FloatTensor(list(map(float, tmp[1:])))
        elif False:
            hit_cnt = 0
            with open(parent_path+'/MAG-CS/computer_science.terms.embed', 'r') as IN:
                num_nodes, num_dim = IN.readline().strip().split(' ')
                init_taxo_emb = torch.zeros((sp_tensor_fos_feature.shape[1]+1, int(num_dim)))
                #IN.readline()
                for i in range(int(num_nodes)-1):
                    tmp = IN.readline().strip().split(' ')
                    if int(tmp[0]) in fos2idx:
                        hit_cnt+=1
                        init_taxo_emb[fos2idx[int(tmp[0])], :] = torch.FloatTensor(list(map(float, tmp[1:])))
        else:
            init_taxo_emb = None
    # embed()
    return sp_tensor_fos_feature, init_taxo_emb, pos_edges


def construct_negative_graph_new(graph, k, etypes, edges=None, num_nodes_dict=None):
    data_dict = {}
    ntypes = set()
    for etype in etypes:
        print("df", etype)
        #etype = graph.to_canonical_etype(etype)
        utype, _, vtype = etype
        ntypes.add(utype)
        ntypes.add(vtype)
        if edges is None:
            src, dst = graph.edges(etype=etype)
        else:
            src, dst = edges[etype]
        neg_src = src.repeat_interleave(k)
        neg_dst = torch.randint(0, graph.number_of_nodes(vtype), (len(src) * k,))
        data_dict[etype] = (neg_src, neg_dst)

    return dgl.heterograph(
        data_dict,
        num_nodes_dict=num_nodes_dict)


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--path', type=str, default='/DATA/EdgeGNN/')
    parser.add_argument('--dataset', type=str, default='MAG', help='MeSH, MAG')
    parser.add_argument('--sim', type=str, default='euclidean')
    parser.add_argument('--loss', type=str, default='nce')
    parser.add_argument('--task', type=str, default='link')
    parser.add_argument('--inference_edge', type=bool, default=False)
    parser.add_argument('--attn', type=bool, default=False)
    parser.add_argument('--test_split', type=float, default=0.1, help='The percentage of test edges')
    parser.add_argument('--gpu', type=int, default=0)
    
    parser.add_argument('--hidden_features', type=int, default=32)
    parser.add_argument('--out_features', type=int, default=128)
    parser.add_argument('--node_embedding_size', type=int, default=128)

    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_negatives', type=int, default=10)
    parser.add_argument('--num_hops', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--test_interval', type=int, default=1)
    parser.add_argument('--tb_log', type=int, default=1, help='whether save tb log')

    parser.add_argument('--bert_paper', type=int, default=0)
    parser.add_argument('--concate_paper', type=int, default=0)
    parser.add_argument('--trainable_paper', type=int, default=0)
    parser.add_argument('--trainable_other', type=int, default=0)
    parser.add_argument('--trainable_rand_paper', type=int, default=0)

    # Venue-year, author-year(pair) later
    parser.add_argument('--year_interval', type=int, default=4)
    parser.add_argument('--venue_year', type=int, default=0)
    parser.add_argument('--author_year', type=int, default=0)
    parser.add_argument('--save_model', type=int, default=1)
    parser.add_argument('--log_type', type=str, default='')
    parser.add_argument('--save_id', type=int, default=0)
    parser.add_argument('--use_bootstrap', type=int, default=0)
    return parser.parse_args()

def evaluate_hit(emb, test_graph,k=100):
    missing_cnt = 0
    if args.sim == 'euclidean':
        t = AnnoyIndex(emb.shape[1], 'euclidean')  # Length of item vector that will be indexed
    else:
        t = AnnoyIndex(emb.shape[1], 'angular')  # Length of item vector that will be indexed
    for i in range(emb.shape[0]):
        t.add_item(i, emb[i,:].tolist())
    t.build(10) # 10 trees
        #t.save('paper_emb_100_1000.ann')
    hit = []
    #src = set(test_graph.edges(etype='citeby')[0].tolist())
    src = set(test_graph.edges(etype='citeby')[0].tolist())
    for i in tqdm(np.random.permutation(list(src)), position=0, leave=True):
    #print(t.get_nns_by_item(i, 10))
    #hit.append(test_graph.has_edges_between([i]*10, AI_T.get_nns_by_item(i, 10), etype='citeby').sum().item())
        hit_count = test_graph.has_edges_between([i]*k, t.get_nns_by_item(i, k), etype='citeby').max().item()
        hit.append(hit_count)
        #if hit_count > 0:
        #    hit.append(1)
        #else:
        #    hit.append(0)
    return np.mean(hit)
    #break

class NCE_HINGE(nn.Module):
    """docstring for NCE_HINGE"""
    def __init__(self, margin=1):
        super(NCE_HINGE, self).__init__()
        self.margin = margin

    def forward(self, scores, others=None):
        #print(scores.shape)
        return torch.sum(F.relu(scores[:, 0].unsqueeze(1) - scores[:, 1:] + self.margin)) / scores.shape[0] + torch.sum(F.relu(scores[:, 0] - 1)) / scores.shape[0]

class NT_XENT(nn.Module):
    def __init__(self):
        super(NT_XENT, self).__init__()
    def forward(self, scores):
        prob = torch.exp(scores[:, 0]) / (torch.exp(scores[:, 1:]).sum(dim=1))
        return -prob.log().sum()

def sampling_edge_inference(model, test_dataloader):
    test_pred, test_label = None, None
    pos_pred, neg_pred = None, None
    for input_nodes, positive_graph, negative_graph, blocks in test_dataloader:
        blocks = [b.to(device) for b in blocks]
        positive_graph = positive_graph.to(device)
        negative_graph = negative_graph.to(device)
        #input_features = blocks[0].srcdata['features']
        
        pos_score, neg_score, reg_loss = model(positive_graph, negative_graph, blocks, inputs, input_weights,degree_scalar)
        
        if args.sim == 'euclidean':
            pos_score, neg_score = - pos_score, - neg_score

        if pos_pred is None:
            pos_pred = pos_score
            neg_pred = neg_score
        else:
            pos_pred = torch.cat((pos_pred, pos_score), dim=0)
            neg_pred = torch.cat((neg_pred, neg_score), dim=0)
        if test_pred is None:
            test_pred = torch.cat((torch.sigmoid(pos_score),torch.sigmoid(neg_score)), dim=0)
            test_label = torch.cat((torch.ones_like(pos_score), torch.zeros_like(neg_score)), dim=0)
        else:
            test_pred = torch.cat((test_pred, torch.sigmoid(pos_score),torch.sigmoid(neg_score)), dim=0)
            test_label = torch.cat((test_label, torch.ones_like(pos_score),torch.zeros_like(neg_score)), dim=0)
        
    roc, f1, auc_ = evaluate(test_pred.cpu(), test_label.cpu())
    #evaluator.eval_metric='hits@100'
    #hits_100 = evaluator.eval({
    #    'y_pred_pos': pos_pred.view(-1),
    #    'y_pred_neg': neg_pred.view(-1),
    #})['hits@100']
    evaluator.eval_metric='mrr'
    mrr = evaluator.eval({
        'y_pred_pos': pos_pred.view(-1),
        'y_pred_neg': neg_pred.reshape(pos_pred.shape[0],NUM_NEGATIVE),
    })['mrr_list']
    print('Test ROC_AUC:', roc, '\tF1', f1, '\tAUC', auc_)
    print(f'MRR:{np.mean(mrr.tolist())}')
    print('inference time:{}'.format(time() - t1))



if __name__ == '__main__':
    #output_taxon(sys.argv[1], sys.argv[2], sys.argv[3])
    args = parse_args()
    args.data_path = dir_path + '/dataset/' + args.dataset

    NUM_NEGATIVE = 5
    TEST_SPLIT = 1.0
    HIDDEN_CHANNELS = args.out_features
    OUT_FEATURES = args.out_features
    NODE_EMBEDDING_SIZE=100
    BATCH_SIZE=256
    LR=1e-3
    args.NUM_NEGATIVE = NUM_NEGATIVE
    args.TEST_SPLIT = TEST_SPLIT
    args.HIDDEN_CHANNELS = HIDDEN_CHANNELS
    args.OUT_FEATURES = OUT_FEATURES
    args.NODE_EMBEDDING_SIZE = NODE_EMBEDDING_SIZE
    args.BATCH_SIZE = BATCH_SIZE
    args.LR = LR

    #embed()
    t0 = time()
    if False:
        data_dump= construct_hetergraph()
        pickle.dump(data_dump, open('dataset/MAG/data_dump.pkl', 'wb'))
    else:
        data_dump = pickle.load(open('dataset/{}/data_dump.pkl'.format(args.dataset), 'rb'))
    t1 = time()
    print(t1-t0)
    g,  yearly_eid_set, p_idx2id_dict, v_name2idx_dict, a_idx2id_dict, p_id2v_name, paper_year_dict = data_dump
    #pid->taxonomy labels
    sp_tensor_fos_feature, init_taxo_emb, pos_edges  = read_taxon_features(args.data_path)
    pos_edges = torch.LongTensor(pos_edges)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else "cpu")

    test_type = ["citeby"]

    etypes = [g.to_canonical_etype(etype) for etype in g.etypes]
    test_type = [g.to_canonical_etype(etype) for etype in test_type]

    # Split edge set for training and testing (Qi: Per type)
    edge_dict, train_edge,test_edge = {}, {} ,{}
    train_dict = {}
    reverse_dict = {}
    for etype in etypes:
        if etype[1] == 'haspaperin':
            continue
        u, v = g.edges(etype=etype)
        edge_dict[etype] = (u,v)
        #edge_dict[etype] = (v, u)
        u_type, _e_type, v_type = etype
        if etype in test_type:
            eids = np.arange(g.number_of_edges(etype=etype))
            eids = np.random.permutation(eids)
            test_size = int(len(eids)*TEST_SPLIT)
            test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
            train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]
            train_edge[etype] = (train_pos_u, train_pos_v)
            train_dict[etype] = eids[test_size:]
            edge_dict[etype] = (test_pos_u, test_pos_v)
            test_edge[etype] = (test_pos_u, test_pos_v)
            reverse_dict[etype] = (v_type, _e_type + '_reverse', u_type)
            reverse_dict[(v_type, _e_type + '_reverse', u_type)] = etype
            #test_dict[etype] = eids[test_size:]
        else:
            train_edge[etype] = (u,v)
            train_edge[(v_type, _e_type + '_reverse', u_type)] = (train_edge[etype][1], train_edge[etype][0])
            edge_dict[(v_type, _e_type + '_reverse', u_type)] = (edge_dict[etype][1], edge_dict[etype][0])


    #training_seeds = pickle.load(open('top_50k_pairs.p', 'rb'))[:1400]
    if 'MAG' in args.data_path:
        seed_path = args.data_path+'/boostrapping_topdegree_50000.p'
        # '/original_top_50k_pairs.p'
        # '/shared/data2/bowenj4/qi/TaxoGNN/boostrapping_topdegree_50000.p'
        # '/shared/data2/bowenj4/qi/TaxoGNN/selected_author_year_1000.p'
        # '/shared/data2/bowenj4/qi/TaxoGNN/selected_author_year_diffauthor.p'
    else:
        if args.use_bootstrap:
            seed_path = args.data_path+'/MeSH_training_seed_bootstrapped.p'
        else:
            seed_path = args.data_path+'/MeSH_training_seed.p'
    training_seeds = pickle.load(open(seed_path, 'rb'))

    train_edge[('paper','citeby', 'paper')] = (torch.LongTensor(training_seeds)[:,0], torch.LongTensor(training_seeds)[:,1])
    # add self-loop
    train_edge[('paper','paper_selfloop', 'paper')] = (np.arange(g.number_of_nodes('paper')), np.arange(g.number_of_nodes('paper')))
    train_edge[('author','author_selfloop', 'author')] = (np.arange(g.number_of_nodes('author')), np.arange(g.number_of_nodes('author')))
    #test_edge[('paper','paper_selfloop', 'paper')] = (np.arange(g.number_of_nodes('paper')), np.arange(g.number_of_nodes('paper')))
    edge_dict[('paper','paper_selfloop', 'paper')] = (np.arange(g.number_of_nodes('paper')), np.arange(g.number_of_nodes('paper')))
    edge_dict[('author','author_selfloop', 'author')] = (np.arange(g.number_of_nodes('author')), np.arange(g.number_of_nodes('author')))
    
    #edge_dict[('paper','citeby', 'paper')] = (torch.LongTensor(training_seeds)[:,0], torch.LongTensor(training_seeds)[:,1])
    #edge_dict[('paper','citeby', 'paper')] = (torch.cat([torch.LongTensor(training_seeds)[:,0], edge_dict[('paper','citeby', 'paper')][0]]), torch.cat([torch.LongTensor(training_seeds)[:,1], edge_dict[('paper','citeby', 'paper')][1]]))

    num_node_dict = {nt:g.number_of_nodes(ntype=nt) for nt in g.ntypes}
    
    train_graph = dgl.heterograph(train_edge, num_nodes_dict=num_node_dict)
    # neg_train_graph = construct_negative_graph_new(g, NUM_NEGATIVE, test_type, train_edge, num_node_dict).to(device)

    #below are code for whole graph inference
    test_graph = dgl.heterograph(edge_dict, num_nodes_dict=num_node_dict) #.to(device)
    neg_test_graph = construct_negative_graph_new(g, NUM_NEGATIVE, test_type, test_edge, num_node_dict).to(device)

    degree_scalar = {}
    for etype in train_graph.etypes:
        if 'citeby' not in etype or 'sim' not in etype:
            degree_scalar[etype] = {'degree_scalar': train_graph.in_degrees(etype=etype).float().to(device)}

    if 'txt' in args.log_type:
        file_logger = dir_path + '/txt_logs/{}_Split_{}_lr_{}_bz_{}.txt'.format(args.dataset, args.TEST_SPLIT, args.LR,
                                                                              args.BATCH_SIZE)
    elif 'tb' in args.log_type:
        pass
    #train_g = dgl.heterograph(edge_dict, num_nodes_dict=num_node_dict)
    eids = np.arange(train_graph.number_of_edges(etype='citeby'))
    #train_dict[('paper','citeby', 'paper')] = eids[test_size:]
    train_dict[('paper','citeby', 'paper')] = eids
    #embed()
    sampler = dgl.dataloading.MultiLayerNeighborSampler([{'writtenby':10, 'paper_selfloop':1, 'author_selfloop':1, 'citeby':0, 'publishedin':10, 'writtenby_reverse':0, 'publishedin_reverse':0}, {'writtenby':0, 'paper_selfloop':1, 'citeby':0, 'publishedin':0, 'author_selfloop':1,'writtenby_reverse':10, 'publishedin_reverse':1}])
    
    test_sampler = dgl.dataloading.MultiLayerNeighborSampler([{'writtenby':-1, 'paper_selfloop':1, 'author_selfloop':1, 'citeby':0, 'publishedin':-1, 'writtenby_reverse':0, 'publishedin_reverse':0}, {'writtenby':0, 'paper_selfloop':1, 'citeby':0, 'publishedin':0, 'author_selfloop':1,'writtenby_reverse':10, 'publishedin_reverse':1}])
    
    #sampler = dgl.dataloading.MultiLayerNeighborSampler([{'paper_selfloop':1, 'writtenby':0, 'citeby':0, 'publishedin':0, 'writtenby_reverse':0, 'citeby_reverse':0, 'publishedin_reverse':0}])
    #sampler = dgl.dataloading.MultiLayerNeighborSampler([{'paper_selfloop':-1, 'writtenby':-1, 'citeby':-1, 'publishedin':-1, 'writtenby_reverse':0, 'citeby_reverse':0, 'publishedin_reverse':0}])
    neg_sampler = dgl.dataloading.negative_sampler.Uniform(NUM_NEGATIVE)
    #eid_dict = {et:g.edges(form='eid', etype=et) for et in g.etypes}
    dataloader = dgl.dataloading.EdgeDataLoader(train_graph, train_dict, sampler,
        exclude='reverse_types', batch_size=BATCH_SIZE,
        negative_sampler=neg_sampler, reverse_etypes = reverse_dict,
        shuffle=True, drop_last=False, num_workers=4)
    
    train_graph = train_graph.to(device)
    

    # if 
    if args.inference_edge:
        test_dataloader = dgl.dataloading.EdgeDataLoader(test_graph, {('paper','citeby', 'paper'):np.arange(test_graph.number_of_edges(etype='citeby') )}, sampler,
            exclude='reverse_types', batch_size=20 * BATCH_SIZE,
            negative_sampler=neg_sampler, reverse_etypes = reverse_dict,
            shuffle=False, drop_last=False, num_workers=4)
    else:
        test_dataloader = dgl.dataloading.NodeDataLoader(
            test_graph, {'paper': np.arange(g.number_of_nodes(ntype='paper'))}, sampler,
            batch_size=20 * BATCH_SIZE,
            shuffle=False,
            drop_last=False,
            num_workers=4)
    test_graph = test_graph.to(device)
    #
    init_kw_emb, init_taxo_emb = None, None
    if args.dataset == 'MAG':
        init_kw_emb = pickle.load(open('dataset/MAG/keywords.emb', 'rb'))
    if args.dataset == 'MAG':
        lm_emb = pickle.load(open(dir_path + '/dataset/MAG/bert_emb.p', 'rb')).to(device)
    else:
        lm_emb = pickle.load(open(dir_path + '/dataset/MeSH_paper_embedding.pkl', 'rb')).to(device)
    taxo_data = pickle.load(open(args.data_path+'/simple_{}_fos_idx_weight.p'.format(args.dataset.lower()), 'rb'))
    taxo_idx, taxo_weight = taxo_data['idx'].int().to(device), taxo_data['weight'].to(device)

    author_data = pickle.load(open(args.data_path+'/{}_author_kw_idx_weight.p'.format(args.dataset.lower()), 'rb'))
    author_idx, author_weight = author_data['idx'].int().to(device), author_data['weight'].to(device)

    model = BatchModel(NODE_EMBEDDING_SIZE, HIDDEN_CHANNELS, OUT_FEATURES, True if args.task == 'link' else False, num_node_dict, {'paper':taxo_idx.max(), 'author':author_idx.max()}, init_embedding={'paper':init_taxo_emb, 'author':init_kw_emb}, bert_embedding=lm_emb,attn=args.attn, device=device).to(device)
    #model.load_state_dict(torch.load('taxograph_ko_1.0_lr_0.001_batchsize_256.pkl'))
    #model.eval()
    if args.loss == 'nce':
        loss_fcn = NCE_HINGE()
    else:
        loss_fcn = NT_XENT()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    evaluator = Evaluator('ogbl-ppa')

    inputs = {'paper': taxo_idx, 'venue': None, 'author': author_idx}
    input_weights = {'paper': taxo_weight, 'venue': None, 'author': author_weight}
    for e in range(args.epochs):
        total_loss = []
        t1 = time()
        #if e % 3 == 0: # and not args.attn:
        if False:
            with torch.no_grad():
                model.eval()
                # if sampling inference
                hits_100 = 0.0
                if args.inference_edge:
                    sampling_edge_inference(model, test_dataloader)
                else:
                    output_emb = None
                    for input_nodes, output_nodes, blocks in test_dataloader:
                        blocks = [b.to(device) for b in blocks]
                        if output_emb is None:
                            output_emb = model.node_inference(input_nodes, blocks, inputs, input_weights, degree_scalar)['paper']
                        else:
                            output_emb = torch.cat((output_emb, model.node_inference(input_nodes, blocks, inputs, input_weights, degree_scalar)['paper']), dim=0)
                    if not args.attn:
                        hits_100 = evaluate_hit(output_emb.cpu(), test_graph.to('cpu'), k=10)
                    print(f'HIT@10 {hits_100}, inference time{time()-t1}')
                    del output_emb
                    torch.cuda.empty_cache()
        model.train()
        for input_nodes, positive_graph, negative_graph, blocks in dataloader:

            blocks = [b.to(device) for b in blocks]
            positive_graph = positive_graph.to(device)
            negative_graph = negative_graph.to(device)
            #input_features = blocks[0].srcdata['features']
            pos_score, neg_score, reg_loss = model(positive_graph, negative_graph, blocks, inputs, input_weights,degree_scalar)
            
            preds = torch.cat([pos_score.unsqueeze(1), neg_score.reshape(pos_score.shape[0], NUM_NEGATIVE)], dim=1)
            #label = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)], dim=0)
            #loss = F.binary_cross_entropy_with_logits(preds, label) # + reg_loss
            loss = loss_fcn(preds)  #+ reg_loss
            #loss = reg_loss
            total_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            #nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
        print("epoch:{}, time:{}, loss:{}".format(e, time() - t1, np.mean(total_loss)))

    torch.save(model.state_dict(), f'taxograph_ko_{TEST_SPLIT}_lr_{LR}_batchsize_{BATCH_SIZE}_{"attn" if args.attn else "no_attn"}.pkl')
    #model.load_state_dict(torch.load(f'taxograph_ko_{TEST_SPLIT}_lr_{LR}_batchsize_{BATCH_SIZE}_{"attn" if args.attn else "no_attn"}.pkl'))
    if args.attn:
        with torch.no_grad():
            model.eval()
            # if sampling inference
            if args.inference_edge:
                sampling_edge_inference(model, test_dataloader)
            else:
                output_emb = None
                for input_nodes, output_nodes, blocks in test_dataloader:
                    blocks = [b.to(device) for b in blocks]
                    if output_emb is None:
                        output_emb = model.node_inference(input_nodes, blocks, inputs, input_weights, degree_scalar)['paper']
                    else:
                        output_emb = torch.cat((output_emb, model.node_inference(input_nodes, blocks, inputs, input_weights, degree_scalar)['paper']), dim=0)
                #hits_100 = evaluate_hit(output_emb.cpu(), test_graph.to('cpu'), k=10)
                #print(f'HIT@10 {hits_100}, inference time{time()-t1}')
                embed()
                pickle.dump({'paper':output_emb.cpu()}, open(f'attn_taxograph_embedding_{args.out_features}.p', 'wb'))
    else:
        with torch.no_grad():
            model.eval()
            pos_pred, neg_pred, output_emb, inner_emb = model.predict(train_graph, test_graph, neg_test_graph, inputs, input_weights,degree_scalar)

            hits_100 = evaluate_hit(output_emb['paper'].cpu(), test_graph.to('cpu'), k=10)
            #hits_100 = 0
            if args.sim == 'euclidean':
                pos_pred, neg_pred = - pos_pred, - neg_pred
                #test_pred = torch.cat([torch.sigmoid(-pos_pred), torch.sigmoid(-neg_pred)]).cpu()

            test_pred = torch.cat([torch.sigmoid(pos_pred), torch.sigmoid(neg_pred)]).cpu()
            test_label = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)]).cpu()

            roc, f1, auc_ = evaluate(test_pred, test_label)
            #evaluator.eval_metric='hits@100'
            #hits_100 = evaluator.eval({
            #    'y_pred_pos': pos_pred.view(-1),
            #    'y_pred_neg': neg_pred.view(-1),
            #})['hits@100']
            evaluator.eval_metric='mrr'
            mrr = evaluator.eval({
                'y_pred_pos': pos_pred.view(-1),
                'y_pred_neg': neg_pred.reshape(pos_pred.shape[0],NUM_NEGATIVE),
            })['mrr_list']
            #
            print('Test ROC_AUC:', roc, '\tF1', f1, '\tAUC', auc_)
            print(f'Test HITS@10:{hits_100} MRR:{np.mean(mrr.tolist())}')
            #embed()
            #embed()
        pickle.dump({'paper':output_emb['paper'].cpu()}, open(f'dataset/{args.dataset}/no_attn_taxograph_embedding_{args.task}_{args.out_features}_{args.save_id}.p', 'wb'))
    # add a line to dump author and paper embeddings
