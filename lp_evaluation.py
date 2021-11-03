import dgl
import pickle
from model.model import BatchHeteroDotProductPredictor
from annoy import AnnoyIndex
from tqdm import tqdm
import numpy as np
from IPython import embed
import argparse
import os
import random
import torch

from accurate_nn import generate_rerank_dict

from ogb.linkproppred import Evaluator
from sklearn.metrics import (auc, f1_score, precision_recall_curve, roc_auc_score, ndcg_score)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='taxognn', help='taxognn, bert, graphsage_ssl, graphsage_unsup')
    parser.add_argument('--dataset', type=str, default='MeSH', help='MeSH, MAG')
    parser.add_argument('--num_negatives', type=int, default=10)
    # parser.add_argument('--test_dict', type=str, default='dgi_testdict_tree30_len1000_rerank')
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--tree', type=int, default=30)
    parser.add_argument('--seed', type=int, default=2077)
    parser.add_argument('--test_dict_len', type=int, default=1000)
    return parser.parse_args()


def evaluate(pred, label):
    pred = np.array(pred).reshape(-1, )
    label = np.array(label, dtype=bool).reshape(-1, )
    sorted_pred = sorted(pred, reverse=True)
    sorted_label = sorted(label, reverse=True)
    true_num = np.sum(label)
    threshold = sorted_pred[true_num]

    y_pred = np.zeros(len(pred), dtype=bool)

    # y_pred[pred>=0.5] = True
    y_pred[pred >= threshold] = True
    # embed()
    ps, rs, _ = precision_recall_curve(label, pred)
    pred_result = y_pred == label
    return roc_auc_score(label, pred), f1_score(label, y_pred), auc(rs, ps)


def construct_negative_graph_new(graph, k, etypes, edges=None, num_nodes_dict=None):
    data_dict = {}
    ntypes = set()
    for etype in etypes:
        print("df", etype)
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

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)
    parent_path = f'/shared/data2/bowenj4/qi/sup-link/{args.dataset}'

    data_dump = pickle.load(open(f'/shared/data2/qiz3/taxograph/dataset/{args.dataset}/data_dump.pkl', 'rb'))
    g, yearly_eid_set, p_idx2id_dict, v_name2idx_dict, a_idx2id_dict, p_id2v_name, paper_year_dict = data_dump
    paper_paper = pickle.load(open(f'/shared/data2/qiz3/taxograph/dataset/{args.dataset}/paper_citeby.pkl', 'rb'))

    all_roc = []
    all_f1 = []
    all_auc = []
    all_hit = []
    all_prec = []
    all_recall = []
    # all_ndcg = []
    all_mrr = []
    qi_path = '/shared/data2/qiz3/taxograph'
    for exp_id in range(5):

        num_node_dict = {nt: g.number_of_nodes(ntype=nt) for nt in g.ntypes}
        if args.method == 'taxognn':
            if args.dataset == 'MAG':
                initial_emb = pickle.load(
                    open(f'{qi_path}/dataset/{args.dataset}/no_attn_taxograph_embedding_node_link_{exp_id}.p', 'rb'))
            else:
                initial_emb = pickle.load(
                    open(f'{qi_path}/dataset/{args.dataset}/no_attn_taxograph_embedding_link_128_{exp_id}.p', 'rb'))
        elif args.method == 'bert':
            if args.dataset == 'MAG':
                lm_emb = pickle.load(open(qi_path + '/dataset/MAG/bert_emb.p', 'rb'))
            else:
                lm_emb = pickle.load(open(qi_path + '/dataset/MeSH_paper_embedding.pkl', 'rb'))
            initial_emb = {'paper': lm_emb}
        #     elif args.method == 'graphsage_ssl':
        #         paper_emb = pickle.load(open(f'{qi_path}/dataset/{args.dataset}/graphsage_embedding_link_0_128.p', 'rb')) # {'paper':pickle.load(open(f'{qi_path}/dataset/{args.dataset}/graphsage_embedding_link_0_128.p', 'rb'))}
        #     elif args.method == 'graphsage_unsup':
        #         paper_emb = pickle.load(open(f'{qi_path}/dataset/{args.dataset}/unsup_graphsage_embedding_link_0_128.p', 'rb'))

        elif args.method == 'graphsage_ssl':
            initial_emb = pickle.load(open(f'{qi_path}/dataset/{args.dataset}/graphsage_embedding_link_{exp_id}_128.p', 'rb'))
        elif args.method == 'graphsage_unsup':
            initial_emb = pickle.load(open(f'{qi_path}/dataset/{args.dataset}/unsup_graphsage_embedding_link_{exp_id}_128.p', 'rb'))
        elif args.method == 'dgi':
            initial_emb = {'paper':torch.load(f'/shared/data2/qiz3/taxograph/dataset/{args.dataset}/{args.method}_emb_node_{exp_id}.p').to('cpu')}
        else:
            initial_emb = {'paper':pickle.load(open(f'/shared/data2/qiz3/taxograph/dataset/{args.dataset}/{args.method}_emb_link_{exp_id}.p','rb'))}

        # test_dict = pickle.load(open(f'/shared/data2/qiz3/taxograph/dataset/{args.dataset}/test_dict/{args.test_dict}.pkl', 'rb'))
        if args.method == 'bert': # emb, tree, paper_paper, parent_path, emb_model, test_dict_len
            test_dict = generate_rerank_dict(initial_emb['paper'], args.tree, paper_paper, parent_path, args.method, args.test_dict_len)
        else:
            test_dict = generate_rerank_dict(initial_emb['paper'], args.tree, paper_paper, parent_path, args.method, args.test_dict_len)

        node_neightbour = {}
        for pair in paper_paper:
            if pair[0] not in node_neightbour:
                node_neightbour[pair[0]] = set()
            node_neightbour[pair[0]].add(pair[1])

        # src = set(test_graph.edges(etype='citeby')[0].tolist())
        src = set(g.edges(etype='citeby')[0].tolist())
        hit = []
        precision = []
        recall = []
        # ndcg = []
        for i in tqdm(list(src), position=0, leave=True):
            hit_count = 1 if len(node_neightbour[i] & set(test_dict[i][:args.k])) > 0 else 0
            tmp_precision = len(node_neightbour[i].intersection(set(test_dict[i][:args.k])))/ args.k
            tmp_recall = len(node_neightbour[i].intersection(set(test_dict[i][:args.k])))/ len(node_neightbour[i])
            ndcg_ground = np.array([1 if test_dict[i][j] in node_neightbour[i] else 0 for j in range(args.k)])
            # tmp_ndcg = ndcg_score(ndcg_ground, np.arange(args.k))
            # hit_count = g.has_edges_between([i] * args.k, t.get_nns_by_item(i, args.k), etype='citeby').max().item()
            precision.append(tmp_precision)
            hit.append(hit_count)
            recall.append(tmp_recall)
            # ndcg.append(tmp_ndcg)
        # print(f'hit@{args.k}:{np.mean(hit)} | precision@{args.k}:{np.mean(precision)} | recall@{args.k}:{np.mean(recall)} | NDCG@{args.k}:{np.mean(ndcg)}')
        print(f'hit@{args.k}:{np.mean(hit)} | precision@{args.k}:{np.mean(precision)} | recall@{args.k}:{np.mean(recall)}')

        evaluator = Evaluator('ogbl-ppa')
        match_model = BatchHeteroDotProductPredictor(initial_emb['paper'].shape[1])
        match_model.eval()
        edge_dict = {('paper', 'citeby', 'paper'): g.edges(etype='citeby')}
        test_graph = dgl.heterograph(edge_dict, num_nodes_dict=num_node_dict)  # .to(device)
        neg_test_graph = construct_negative_graph_new(g, args.num_negatives, [('paper', 'citeby', 'paper')], edge_dict, num_node_dict)

        # because the model output distance not similarity, negate it
        pos_pred = - match_model(test_graph, initial_emb,False,args.method in ['dgi'])
        neg_pred = - match_model(neg_test_graph, initial_emb,False,args.method in ['dgi'])

        test_pred = torch.cat([torch.sigmoid(pos_pred), torch.sigmoid(neg_pred)]).cpu()
        test_label = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)]).cpu()

        roc, f1, auc_ = evaluate(test_pred, test_label)
        evaluator.eval_metric = 'mrr'
        mrr = evaluator.eval({
            'y_pred_pos': pos_pred.view(-1),
            'y_pred_neg': neg_pred.reshape(pos_pred.shape[0], args.num_negatives),
        })['mrr_list']
        #
        print('Test ROC_AUC:', roc, '\tF1', f1, '\tAUC', auc_)
        print(f'Test HITS@10:{np.mean(hit)} MRR:{np.mean(mrr.tolist())}')

        all_roc.append(roc)
        all_f1.append(f1)
        all_auc.append(auc_)
        all_hit.append(np.mean(hit))
        all_prec.append(np.mean(precision))
        all_recall.append(np.mean(recall))
        # all_ndcg.append(np.mean(ndcg))
        all_mrr.append(np.mean(mrr.tolist()))

    print()
    # print(f'hit@{args.k}:{np.mean(all_hit), np.std(all_hit)} | precision@{args.k}:{np.mean(all_prec), np.std(all_prec)}  \
    #         | recall@{args.k}:{np.mean(all_recall), np.std(all_recall)} | NDCG@{args.k}:{np.mean(all_ndcg), np.std(all_ndcg)} | MRR:{np.mean(all_mrr), np.std(all_mrr)}')
    print(f'hit@{args.k}:{np.mean(all_hit), np.std(all_hit)} | precision@{args.k}:{np.mean(all_prec), np.std(all_prec)}  \
            | recall@{args.k}:{np.mean(all_recall), np.std(all_recall)} | MRR:{np.mean(all_mrr), np.std(all_mrr)}')
    print('Test ROC_AUC:', np.mean(all_roc), np.std(all_roc), '\tF1', np.mean(all_f1), np.std(all_f1), '\tAUC', np.mean(all_auc), np.std(all_auc))
