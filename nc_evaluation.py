import argparse
import gzip
import json
import os
import pickle
import random

import torch
from sklearn.metrics import f1_score
from torch import nn, optim
import torch.utils.data as tdata
from data_loader.dataset import TaxonDataset
import torch.nn.functional as F
import numpy as np
import time

# dir_path = os.path.dirname(os.path.realpath(__file__))
# from IPython import embed
# # dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = '/shared/data2/qiz3/taxograph'


def set_seed(seed):
    if seed is None:
        seed = int(time.time())
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def data_processing(dataset, data_src_path):
    ###
    # 1.check how many papers include layer-1 fos & how many papers only have one layer-1
    # 2.make fos to node label, save
    # 3.write the eval pipeline
    ##
    id2label_dict = {}
    with open('/shared/data2/qiz3/taxograph/dataset/{}/id2label.txt'.format(dataset), 'r') as file:
        for line in file:
            fos_id, name = line.strip().split('\t')
            id2label_dict[fos_id] = name

    taxo_dataset = TaxonDataset(dataset, path=data_src_path)
    data_dump = pickle.load(open('/shared/data2/qiz3/taxograph/dataset/{}/data_dump.pkl'.format(dataset), 'rb'))
    g,  yearly_eid_set, p_idx2id_dict, v_name2idx_dict, a_idx2id_dict, p_id2v_name, paper_year_dict = data_dump

    with open(data_src_path + '/paper_fos_score_vec_dict.pkl', 'rb') as handle:
        paper_fos_score_vec_dict = pickle.load(handle)

    with open(data_src_path + '/fos_idx_dict.pkl', 'rb') as handle:
        fos_idx_dict = pickle.load(handle)

    fos_id_2_name = {}
    with open(data_src_path + '/id2label.txt', 'r') as file:
        # fos_idx_dict = pickle.load(handle)
        for line in file:
            fos_id, name = line.strip().split('\t')
            fos_id_2_name[fos_id] = name
    filtered_fos_name = set()
    if dataset == 'MAG':
        with open(data_src_path + '/labels_new.txt', 'r') as file:
            for line in file:
                fos_name = line.strip()
                filtered_fos_name.add(fos_name)
        print('filter fos name num:{}'.format(len(filtered_fos_name)))
    useful_fos_idx_set = set()
    useful_fos_idx_to_label = {}
    label_to_fos_name = {}
    for k, v in fos_idx_dict.items():
        if dataset == 'MAG':
            if fos_id_2_name[str(k)] not in filtered_fos_name:
                continue
        if taxo_dataset.get_depth(k) == 0:
            useful_fos_idx_set.add(v)
            if v not in useful_fos_idx_to_label:
                useful_fos_idx_to_label[v] = len(useful_fos_idx_to_label)
                label_to_fos_name[len(useful_fos_idx_to_label)] = id2label_dict[str(k)]
    print('used fos idx num:{}'.format(len(useful_fos_idx_set)))

    include_level1_fos, only_level1_fos = 0, 0
    p_idx_id_label_tensor = torch.zeros((len(paper_fos_score_vec_dict), 2))
    for p_idx in range(len(paper_fos_score_vec_dict)):
        fos_vec = paper_fos_score_vec_dict[p_idx2id_dict[p_idx]]
        rows, cols = fos_vec.nonzero()
        if len(set(cols).intersection(useful_fos_idx_set)) == 1:
            only_level1_fos += 1
            fos_idx = list(set(cols).intersection(useful_fos_idx_set))[0]
            p_idx_id_label_tensor[p_idx] = torch.tensor([int(p_idx2id_dict[p_idx]), int(useful_fos_idx_to_label[fos_idx])])
            p_idx_id_label_tensor = p_idx_id_label_tensor.long()
        if len(set(cols).intersection(useful_fos_idx_set)) >= 1:
            include_level1_fos +=1

    print('{} level1 fos num:{} inlude:{} only:{}'.format(dataset, len(useful_fos_idx_set), include_level1_fos, only_level1_fos))


    with open(data_src_path + '/p_idx_id_label.pkl', 'wb') as handle:
        pickle.dump(p_idx_id_label_tensor, handle)

    with open(data_src_path + '/node_label_to_fos_name.pkl', 'wb') as handle:
        pickle.dump(label_to_fos_name, handle)

    # MAG level1 fos num:34 inlude:571590 only:262504
    # MeSH level1 fos num:1241 inlude:636308 only:246149
    # MeSH level1 fos num:68 inlude:59700 only:56310



class MLP(torch.nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(nn.Linear(args.input_size, args.hidden_size),
                                    nn.ReLU(), nn.Linear(args.hidden_size, args.class_num))
    def forward(self, x):
        return self.layers(x)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='gat_graphsage', help='gat_graphsage, orig_graphsage, graphsage, dgi, taxognn')
    parser.add_argument('--dataset', type=str, default='MAG', help='MeSH, MAG')
    parser.add_argument('--dim', type=int, default=32)
    parser.add_argument('--num_negatives', type=int, default=5)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--train_ratio', type=float, default=0.05)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--test_interval', type=int, default=5)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--perC_sample', type=int, default=1)
    parser.add_argument('--mode', type=str, default='node')
    parser.add_argument('--save_id', type=int, default=0)
    return parser.parse_args()


def evaluation(preds, labels):
    correct = preds.eq(labels).double()
    correct = correct.sum()
    acc = correct / len(labels)
    marco_f1 = f1_score(labels, preds, average='macro')
    micro_f1 = f1_score(labels, preds, average='micro')
    return acc, marco_f1, micro_f1



def print_sampled_content(label_dump, p_idx):
    with open('/shared/data2/qiz3/taxograph/dataset/MeSH/node_label_to_fos_name.pkl', 'rb') as handle:
        label_to_fos_name = pickle.load(handle)

    p_idx_list = p_idx.tolist()
    sampled_p_idx = random.sample(p_idx_list, 30)
    sampled_p_ids = label_dump[:, 0][sampled_p_idx].tolist()
    sampled_labels = label_dump[:, 1][sampled_p_idx].tolist()

    sampled_p_id_set = set(sampled_p_ids)
    label_json_path = '/shared/data2/qiz3/taxograph/dataset/MeSH/MeSH_Title_Abstract_Year.txt'
    prev_id = 0
    with open(label_json_path, 'r') as file:
        for idx, line in enumerate(file):
            paper_id = int(line.strip().split('\t')[0])
            if paper_id in sampled_p_id_set:
                if paper_id != prev_id:
                    paper_label = sampled_labels[sampled_p_ids.index(paper_id)]
                    print(label_to_fos_name[paper_label])
                print(line.strip())
                prev_id = paper_id


if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)
    args.data_src_path = dir_path + '/dataset/' + args.dataset
    ####
    # data_processing(args.dataset, args.data_src_path)
    # exit(1)
    ####
    data_dump = pickle.load(open(f'/shared/data2/qiz3/taxograph/dataset/{args.dataset}/data_dump.pkl', 'rb'))
    g, yearly_eid_set, p_idx2id_dict, v_name2idx_dict, a_idx2id_dict, p_id2v_name, paper_year_dict = data_dump
    label_dump = pickle.load(open(args.data_src_path + f'/p_idx_id_label.pkl', 'rb'))
    p_idx = label_dump[:,0].nonzero().reshape(-1)
    ####
    # print_sampled_content(label_dump, p_idx)
    # exit(1)
    ####
    labels = label_dump[:,1][p_idx]
    args.class_num = len(labels.unique())
    if args.perC_sample:
        train_idx = []
        for i in range(args.class_num):
            class_ilabel_idxs = list(np.where(labels == i)[0])
            random.shuffle(class_ilabel_idxs)
            train_idx.extend(class_ilabel_idxs[:int(len(class_ilabel_idxs) * args.train_ratio) + 1])
        test_idx = list(set(list(range(p_idx.shape[0]))) - set(train_idx))
    else:
        train_idx = random.sample(range(p_idx.shape[0]), int(args.train_ratio*p_idx.shape[0]))
        test_idx = list(set(list(range(p_idx.shape[0]))) - set(train_idx))

    train_p_idx, test_p_idx = p_idx[train_idx], p_idx[test_idx]
    train_labels, test_labels = labels[train_idx], labels[test_idx]
    train_dataloader = tdata.DataLoader(tdata.TensorDataset(torch.LongTensor(train_p_idx), torch.LongTensor(train_labels)),
        batch_size=args.batch_size, shuffle=True)
    test_dataloader = tdata.DataLoader(tdata.TensorDataset(torch.LongTensor(test_p_idx), torch.LongTensor(test_labels)),
        batch_size=args.batch_size, shuffle=False)

    num_node_dict = {nt: g.number_of_nodes(ntype=nt) for nt in g.ntypes}
    if args.method == 'taxognn':
        initial_emb = pickle.load(open(f'/shared/data2/qiz3/taxograph/dataset/{args.dataset}/no_attn_taxograph_embedding_node_256.p', 'rb'))
    elif args.method == 'bert':
        initial_emb = {'paper': pickle.load(open(f'/shared/data2/qiz3/taxograph/dataset/{args.dataset}/bert_emb.p', 'rb'))}
    elif args.method == 'graphsaint':
        initial_emb = {'paper': pickle.load(open(f'/shared/data2/qiz3/taxograph/dataset/{args.dataset}/graphsaint_emb.p', 'rb'))}
    elif args.method == 'fame':
        initial_emb = {'paper': pickle.load(open(f'/shared/data2/qiz3/taxograph/dataset/{args.dataset}/fame_emb.p', 'rb'))}
    elif args.method == 'dgi':
        paper_embed = torch.load(f'/shared/data2/qiz3/taxograph/dataset/{args.dataset}/dgi_emb_no_epoch120_0.p', map_location='cpu')
        # initial_emb = {'paper': torch.load(f'/shared/data2/qiz3/taxograph/dataset/{args.dataset}/dgi_emb_node_0.pkl')}
        initial_emb = {'paper': paper_embed}
    elif args.method in ['graphsage', 'orig_graphsage', 'gat_graphsage']:
        initial_emb = pickle.load(open(f'/shared/data2/qiz3/taxograph/dataset/{args.dataset}/{args.method}_embedding_{args.mode}_{args.save_id}_128.p', 'rb'))
    else:
        initial_emb = pickle.load(open(f'dataset/{args.dataset}/{args.method}_embedding_node.p', 'rb'))

    print('Embedding size: {}'.format(initial_emb['paper'].shape[1]))
    args.input_size = initial_emb['paper'].shape[1]
    args.hidden_size = args.dim

    model = MLP(args)
    paper_emb = initial_emb['paper'].cuda()
    model.cuda()
    log_file = os.path.dirname(os.path.realpath(__file__)) + '/txt_logs/nc_{}_{}_{}_train_ratio_{}.txt'.format(args.method, args.dataset, args.save_id, args.train_ratio)


    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_acc, best_marco_f1, best_mirco_f1 = 0, 0, 0
    best_reult_log = "Acc:{}, Marco-F1:{}, Micro-F1:{}".format(0,0,0)

    for i in range(args.epochs):
        total_loss  = []
        model.train()
        for p_idx, label in train_dataloader:
            logit = model(paper_emb[p_idx])
            log_pred = F.log_softmax(logit, dim=1)
            loss_train = F.nll_loss(log_pred, label.cuda())
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            total_loss.append(loss_train.item())
        print("Epoch:{} Train Loss:{}".format(i, np.mean(total_loss)))

        if (i+1) % args.test_interval == 0:
            label_list = []
            pred_list = []
            with torch.no_grad():
                model.eval()
                for p_idx, label in test_dataloader:
                    logit = model(paper_emb[p_idx])
                    preds = logit.max(1)[1].type_as(label)
                    label_list.append(label)
                    pred_list.append(preds)

            acc, marco_f1, micro_f1 = evaluation(torch.cat(pred_list), torch.cat(label_list))

            if acc > best_acc:
                best_acc = acc
                best_marco_f1 = marco_f1
                best_mirco_f1 = micro_f1
                best_reult_log = "Acc:{}, Marco-F1:{}, Micro-F1:{}".format(best_acc, best_marco_f1, best_mirco_f1)
            print("Acc:{}, Marco-F1:{}, Micro-F1:{}".format(acc, marco_f1, micro_f1))

    with open(log_file, 'a+') as file:
        file.write('{} {}\n'.format(best_marco_f1, best_mirco_f1))