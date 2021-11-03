import torch as th
import torch.nn as nn
import math
import numpy as np
import dgl.nn as dglnn
from tqdm import tqdm

import dgl
import dgl.function as fn
from dgl.utils import expand_as_pair 
from dgl.nn import HeteroGraphConv
import dgl.nn as dglnn
import torch.nn.functional as F
import torch
from torch.nn.parameter import Parameter

from collections import Counter
import argparse

import pickle
from annoy import AnnoyIndex

NUM_NEGATIVE = 5
TEST_SPLIT = 0.95
HIDDEN_CHANNELS = 32
OUT_FEATURES = 32
NODE_EMBEDDING_SIZE=64
BATCH_SIZE=256
LR=1e-2

# some parameters you should add here!
parser = argparse.ArgumentParser()
parser.add_argument('--degree_top', type=int, default=20000)
parser.add_argument('--final_num', type=int, default=10000)
parser.add_argument('--nearest_neighbour_num', type=int, default=3)
parser.add_argument('--author_threshold1', type=float, default=0.2, help='xxx')
parser.add_argument('--author_threshold2', type=float, default=0.5, help='xxxx')
parser.add_argument('--year_range', type=int, default=5, help='xxxxx')
parser.add_argument('--model_embedding', type=str, default='taxograph_bootstrapping_start1000.p', help='xxxxx')
parser.add_argument('--model_seeds', type=str, default='selected_author_year_1000.p', help='xxxxx')
parser.add_argument('--text_data', type=str, default='dataset/MAG/MAG_CS_Title_Abstract_Year.txt', help='xxxxx')
parser.add_argument('--save_dir', type=str, default='boostrapping_topdegree_all_random.p', help='xxxxx')


my_parser =  parser.parse_args()

## Read Graph
from preprocess.taxo_dgl_construction import construct_hetergraph, sparse_mx_to_torch_sparse_tensor
from time import time
t0 = time()
g,  _, p_idx2id_dict, _, _, _, _= construct_hetergraph()
t1 = time()
#pickle.dump(g, open('dataset/mag_cs_0726_g.pkl', 'wb'))
print(t1-t0)
p_id2idx_dict = {v:k for k,v in p_idx2id_dict.items()}

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


test_type = ["citeby"]

etypes = [g.to_canonical_etype(etype) for etype in g.etypes]
test_type = [g.to_canonical_etype(etype) for etype in test_type]

# Split edge set for training and testing (Qi: Per type)
edge_dict, test_edge, train_edge = {}, {}, {}
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
        test_edge[etype] = (test_pos_u, test_pos_v)
        reverse_dict[etype] = (v_type, _e_type + '_reverse', u_type)
        reverse_dict[(v_type, _e_type + '_reverse', u_type)] = etype
        #test_dict[etype] = eids[test_size:]
    else:
        train_edge[etype] = (u,v)
    train_edge[(v_type, _e_type + '_reverse', u_type)] = (train_edge[etype][1], train_edge[etype][0])
    edge_dict[(v_type, _e_type + '_reverse', u_type)] = (edge_dict[etype][1], edge_dict[etype][0])

num_node_dict = {nt:g.number_of_nodes(ntype=nt) for nt in g.ntypes}
train_edge[('paper','paper_selfloop', 'paper')] = (np.arange(g.number_of_nodes('paper')), np.arange(g.number_of_nodes('paper')))
#test_edge[('paper','paper_selfloop', 'paper')] = (np.arange(g.number_of_nodes('paper')), np.arange(g.number_of_nodes('paper')))
edge_dict[('paper','paper_selfloop', 'paper')] = (np.arange(g.number_of_nodes('paper')), np.arange(g.number_of_nodes('paper')))


train_graph = dgl.heterograph(train_edge, num_nodes_dict=num_node_dict)
neg_train_graph = construct_negative_graph_new(g, NUM_NEGATIVE, test_type, train_edge, num_node_dict)

test_graph = dgl.heterograph(test_edge, num_nodes_dict=num_node_dict)
neg_test_graph = construct_negative_graph_new(g, NUM_NEGATIVE, test_type, test_edge, num_node_dict)


# construct paper text dict
p_id2text_dict = {}
with open(my_parser.text_data) as IN:
    IN.readline()
    for line in IN:
        tmp = line.strip().split('\t')
        if len(tmp) > 4 and tmp[4] == 'Title':
            if tmp[0] not in p_id2text_dict:
                p_id2text_dict[tmp[0]] = tmp[1] + '___' + tmp[6]
                
            if tmp[0] == '1975335030':
                print(tmp)
            if tmp[0] == '2993424428':
                print(tmp)

citeby_indegree = g.in_degree(v='__ALL__', etype='citeby')
print(citeby_indegree)

# search for top degree nodes
topk = my_parser.degree_top

topk_values, topk_indices = torch.topk(citeby_indegree, topk)
topk_indices = topk_indices.numpy().tolist()
# print(topk_values, topk_indices)


# load init embeddings
taxo_emb = pickle.load(open(my_parser.model_embedding,'rb'))
taxo_ko = taxo_emb['paper']

training_sample = pickle.load(open(my_parser.model_seeds, 'rb'))

# construct nearest neighbour index
AI_T = AnnoyIndex(taxo_ko.shape[1], 'euclidean')  # Length of item vector that will be indexed

for i in tqdm(range(taxo_ko.shape[0])):
    v = taxo_ko[i].numpy().tolist()
    AI_T.add_item(i, v)
AI_T.build(20) # 10 trees


# search in nearest neighbour
l_topk,r_topk = [], []
training_degree_seeds2 = []
K = my_parser.nearest_neighbour_num # number of nearest neighbour
author_threshold1 = my_parser.author_threshold1
author_threshold2 =  my_parser.author_threshold2
year_range = my_parser.year_range 

for i in tqdm(topk_indices):
    tmp_seeds = []
    knn_i = AI_T.get_nns_by_item(i, 100)
    for neighbour in knn_i[1:]:
        if len(tmp_seeds) == K:
            training_degree_seeds2 = training_degree_seeds2 + tmp_seeds
            break
        else:
            _, author1 = g.out_edges(i, etype='writtenby')
            _, author2 = g.out_edges(neighbour, etype='writtenby')
            text1, text2 = p_id2text_dict[str(p_idx2id_dict[i])], p_id2text_dict[str(p_idx2id_dict[neighbour])]
        
            year1, year2 = int(text1.split('___')[0]), int(text2.split('___')[0])
            year_r = abs(year1-year2)  
            author_score = len(set(author1.numpy()) & set(author2.numpy())) / len(set(author1.numpy()) | set(author2.numpy()))
            
            if author_score <= author_threshold2 and author_score >= author_threshold1 and year_r < year_range:
                tmp_seeds.append([i, neighbour])

training_seeds_all = training_degree_seeds2 + training_sample

# rank and search in bootstrapped seeds
ll, rr = [], []
for lid, rid in training_seeds_all:
    ll.append(lid)
    rr.append(rid)

# cos = nn.CosineSimilarity(dim=1, eps=1e-6)
# similarity_score = cos(taxo_ko[ll,:], taxo_ko[rr,:])

# training_seeds_final = []
# for idx in tqdm(similarity_score.argsort(descending=True)[:my_parser.final_num].tolist()):
#     if ll[idx] != rr[idx]:
#         training_seeds_final.append((ll[idx], rr[idx]))

# training_seeds_final = []
# for i in range(my_parser.final_num):
#     training_seeds_final.append((ll[i], rr[i]))

#### dump
pickle.dump(training_seeds_all, open(my_parser.save_dir, 'wb'))
