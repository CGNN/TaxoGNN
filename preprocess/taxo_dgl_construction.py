import os
import pickle

import dgl
import torch
import json
import numpy as np
from scipy import sparse
import dgl.function as fn
from tqdm import tqdm
from IPython import embed
from collections import defaultdict

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(dir_path, os.pardir))
# #parent_path = os.path.abspath(os.path.join(dir_path, os.pardir))
# parent_path = '/shared/data2/qiz3/taxograph/dataset/MAG/'
import sys
sys.path.append(parent_path)
from data_loader.dataset import TaxonDataset
import torch.nn as nn
import torch.nn.functional as F
# label_json_path = parent_path + 'labeled.json'
# fos_score_path = parent_path + '/dataset/fos_nli_score.txt'


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def print_author(data_path, author_mapping_dict, A_matrix):
    author_name, author_score = dict(), defaultdict(int)
    with open(data_path + '/id2author.txt', 'r') as fin:
        for line in fin:
            tmp = line.strip().split('\t')
            author_name[tmp[0]] = tmp[1]
    for idx in author_mapping_dict:
        # same name issue
        author_score[author_name[str(author_mapping_dict[idx])]] = max(A_matrix[idx], author_score[author_name[str(author_mapping_dict[idx])]]) 
    return author_score
        
    
def print_taxo(data_path, fos_dict, venue_mapping_dict, V_matrix):
    with open(data_path + '/idx2name_dict.pkl', 'rb') as handle:
        idx2name_dict = pickle.load(handle)
    #print(venue_mapping_dict)
    #venue_names = venue_mapping_dict.keys()
    #venue_names = ['KDD', 'WWW', 'ICDE', 'WSDM', 'CIKM']
    venue_names = ['CVPR', 'ICCV', 'ECCV', 'MM', 'IROS', 'ICIP', 'NeurIPS', 'ICML', 'ICLR']
    #venue_names = ['NeurIPS', 'ICML', 'ICLR', 'AAAI', 'IJCAI', 'COLT']
    #venue_names = ['SIGMOD']
    venue_top_terms = dict()
    top_taxon = ['computer_engineering','operating_system',
    'library_science','software_engineering',
    'real_time_computing','computer_vision','telecommunications','data_science',
    'knowledge_management','data_mining','computer_architecture','database',
    'natural_language_processing','information_retrieval','machine_learning',
    'human_computer_interaction','embedded_system','distributed_computing',
    'computer_hardware','computer_network','programming_language','artificial_intelligence',
    'parallel_computing','algorithm','speech_recognition','world_wide_web',
    'theoretical_computer_science','computer_security','internet_privacy',
    'multimedia','pattern_recognition','simulation','computer_graphics_images']

    #top_taxon = ['topic_model']
    #print(idx2name_dict[3891])

    if len(V_matrix.shape) == 1:
        vec = V_matrix
        idx_list = list(np.nonzero(V_matrix)[0])
        taxo_score = []
        
        for idx in idx_list:
            if idx in idx2name_dict:
                score = vec[idx]
                name = idx2name_dict[idx]

                if name.replace(' ', '_') in set(top_taxon):
                    taxo_score.append([name, score])

        if len(taxo_score) > 0:
            max_score = max(list(map(lambda x:x[1], taxo_score)))
            #for t in taxo_score:
            #    t[1] /= max_score
            return sorted(taxo_score, key=lambda x:x[1], reverse=True)[:15]
    else:
        print('here')
        for venue_name in venue_names:
            vec = V_matrix[venue_mapping_dict[venue_name], :].reshape(-1,)
            #print(vec)
            idx_list = list(np.nonzero(vec)[0])
            taxo_score = []
            for idx in idx_list:
                if idx in idx2name_dict:
                    score = vec[idx]
                    name = idx2name_dict[idx]

                    if name.replace(' ', '_') in set(top_taxon):
                        taxo_score.append([name, score])

            if len(taxo_score) > 0:
                max_score = max(list(map(lambda x:x[1], taxo_score)))
                #for t in taxo_score:
                #    t[1] /= max_score
                
                venue_top_terms[venue_name] = sorted(taxo_score, key=lambda x:x[1], reverse=True)[:15]
                #venue_top_terms[venue_name] = dict(taxo_score)
        
    return venue_top_terms

def update_ancestor(fos_dict, V_matrix):
    id2idx = {}
    for k,v in fos_dict.items():
        id2idx[str(k)] = v


    taxonomy = TaxonDataset('computer_science', path=parent_path+'/MAG-CS')
    taxo_graph = taxonomy.taxonomy.copy()
    orig_leaves = [v for v, d in taxo_graph.out_degree() if d == 0]
    taxo_graph.remove_nodes_from(orig_leaves)
    leaves = [v for v, d in taxo_graph.out_degree() if d == 0]

    prop_rounds = []
    while len(leaves) != 0:
        current_round = []
        for leaf in leaves:
            current_round.append(leaf.tx_id)
        prop_rounds.append(current_round)
        taxo_graph.remove_nodes_from(leaves)
        leaves = [v for v, d in taxo_graph.out_degree() if d == 0]
        #print(len(leaves))
       
    #print(type(prop_rounds[-2]))
    #print("Num of propagation rounds:{}".format(len(prop_rounds)))



    for id_list in prop_rounds:
        for parent_id in id_list:
            children_id = taxonomy._query_children(parent_id)[1:]
            if parent_id not in id2idx:
                continue
            parent_idx = id2idx[parent_id]
            temp_list = []
            for child_id in children_id:
                if child_id.tx_id in id2idx:
                    temp_list.append(id2idx[child_id.tx_id])
            children_idx = np.array(temp_list)
            if len(children_idx) == 0:
                continue
            # A_matrix[:, parent_idx] = A_matrix[:, children_idx].sum(axis=1)
            #print(V_matrix[0, parent_idx])
            V_matrix[:, parent_idx] += V_matrix[:, children_idx].max(axis=1)
            #print(children_idx, V_matrix[0, children_idx].sum())
    #print(type(prop_rounds[-2][0]))

    return V_matrix, [id2idx[x] for x in prop_rounds[-2]]
    
def construct_hetergraph(data_path, dataset, year_interval=4, venue_year=False, author_year=False):
    # graph_data = None
    # g = dgl.heterograph(graph_data)
    # 1.read all P,V,A nodes; and create a mapping dict
    fos_set = set()
    authors_set = set()
    papers_set, pid_dict = set(), dict()
    venues_set = set()
    label_json_path = data_path + '/labeled.json' if 'MAG' in data_path else data_path + '/MeSH.json'
    with open(label_json_path) as fin:
        for idx, line in enumerate(fin):
            js = json.loads(line)
            fos_list = js['fos'] if dataset == 'MAG' else js['MeSH']
            authors_list = js['author']
            papers_set.add(int(js['paper']))
            #pid_dict[int(js['paper'])] = js['paper']
            venues_set.add(js['venue'])
            for fos in fos_list:
                if dataset == 'MAG':
                    fos_set.add(int(fos))
                else:
                    fos_set.add(fos)
            for author in authors_list:
                authors_set.add(int(author))

    fos_set = sorted(fos_set)
    fos_id2idx_set = {}
    for fos in fos_set:
        fos_id2idx_set[fos] = len(fos_id2idx_set)

    authors_set = sorted(authors_set)
    papers_set = sorted(papers_set)
    venues_set = sorted(venues_set)
    a_id2idx_dict = {}
    p_id2idx_dict = {}
    v_name2idx_dict = {}
    a_idx2id_dict = {}
    p_idx2id_dict = {}
    v_idx2name_dict = {}

    for p in papers_set:
        p_id2idx_dict[p] = len(p_id2idx_dict)
        p_idx2id_dict[p_id2idx_dict[p]] = p
    for a in authors_set:
        a_id2idx_dict[a] = len(a_id2idx_dict)
        a_idx2id_dict[a_id2idx_dict[a]] = a
    for v in venues_set:
        v_name2idx_dict[v] = len(v_name2idx_dict)
        v_idx2name_dict[v_name2idx_dict[v]] = v

    # define edge: p->v: ('paper', 'publishedin', 'venue'); p->a: ('paper', 'writtenby', 'author'); a->v ('author', '$paperidx', 'venue')
    with open(data_path+'/author_paper_list_dict.pkl', 'rb') as handle:
        author_paper_list_dict = pickle.load(handle)
    with open(data_path+'/venue_paper_list_dict.pkl', 'rb') as handle:
        venue_paper_list_dict = pickle.load(handle)
    with open(data_path + '/paper_citation_paper_list_dict.pkl', 'rb') as handle:
        paper_citation_paper_list_dict = pickle.load(handle)
    # with open(parent_path+'/dataset/venue_mapping_dict.pkl', 'rb') as handle:
    #     venue_mapping_dict = pickle.load(handle)
    if  venue_year or author_year:
        with open(data_path + '/paper_year_dict.pkl', 'rb') as handle:
            pre_paper_year_dict = pickle.load(handle)
        year_set = set()
        for k,v in pre_paper_year_dict.items():
            year_set.add(v)
        total_years = len(year_set)
        num_interval = (total_years // year_interval) + 1 if total_years % year_interval else total_years // year_interval

    yearly_eid_set = {'publishedin': defaultdict(list), 'writtenby': defaultdict(list), 'citeby': defaultdict(list), 'haspaperin': defaultdict(list)}
    year = range(2000,2021)
    paper_year_dict = {}
    if dataset == 'MAG':
        year_json_path = data_path+'/MAG_CS_Title_Abstract_Year.txt'.format(dataset)
        with open(year_json_path) as fin:
            fin.readline()
            for line in fin:
                tmp = line.strip().split('\t')
                if int(tmp[1]) <= 2000:
                    #yearly_papers_set[2000].append(tmp[0])
                    paper_year_dict[int(tmp[0])] = 2000
                else:
                    #yearly_papers_set[int(tmp[1])].append(tmp[0])
                    paper_year_dict[int(tmp[0])] = int(tmp[1])
            

    p_id2v_name = {}
    for v, ps in venue_paper_list_dict.items():
        for p in ps:
            p_id2v_name[p] = v

    p_idx2v_idx_edge = []

    for v_name,p_ids in venue_paper_list_dict.items():
        v_idx = v_name2idx_dict[v_name]
        for p_id in p_ids:
            y_id = None if not venue_year and not author_year else year_to_yid(pre_paper_year_dict[p_id], year_interval)
            p_idx = p_id2idx_dict[p_id]
            if dataset == 'MAG':
                yearly_eid_set['publishedin'][paper_year_dict[p_id]].append(len(p_idx2v_idx_edge))
            v_idx_ = v_idx*num_interval + y_id if venue_year else v_idx
            p_idx2v_idx_edge.append((p_idx, v_idx_))

    p_idx2a_idx_edge = []
    for a_id,p_ids in author_paper_list_dict.items():
        a_idx = a_id2idx_dict[a_id]
        for p_id in p_ids:
            y_id = None if not venue_year and not author_year else year_to_yid(pre_paper_year_dict[p_id], year_interval)
            p_idx = p_id2idx_dict[p_id]
            if dataset == 'MAG':
                yearly_eid_set['writtenby'][paper_year_dict[p_id]].append(len(p_idx2a_idx_edge))
            a_idx_ = a_idx*num_interval+y_id if author_year else a_idx
            p_idx2a_idx_edge.append((p_idx, a_idx_))

    a_idx2v_idx_edge = []
    for a_id, p_ids in author_paper_list_dict.items():
        a_idx = a_id2idx_dict[a_id]
        for p_id in p_ids:
            y_id = None if not venue_year and not author_year else year_to_yid(pre_paper_year_dict[p_id], year_interval)
            v_name = p_id2v_name[p_id]
            v_idx = v_name2idx_dict[v_name]
            # might need to specify further
            v_idx_ = v_idx*num_interval+y_id if venue_year else v_idx
            a_idx_ = a_idx*num_interval+y_id if author_year else a_idx
            if dataset == 'MAG':
                yearly_eid_set['haspaperin'][paper_year_dict[p_id]].append(len(a_idx2v_idx_edge))
            a_idx2v_idx_edge.append((a_idx_, v_idx_))
            
    
    p_idx2p_idx_edge = []
    num_missing_key = 0
    for p_id, cite_ids in  paper_citation_paper_list_dict.items():
        if p_id in p_id2idx_dict:
            p_idx = p_id2idx_dict[p_id]
        else:
            num_missing_key += 1
            continue
        for c_id in cite_ids:
            c_idx = p_id2idx_dict[c_id]
            if dataset == 'MAG':
                yearly_eid_set['citeby'][paper_year_dict[c_id]].append(len(p_idx2p_idx_edge))
            p_idx2p_idx_edge.append((p_idx, c_idx))
        
    print('missing key {}'.format(float(num_missing_key) / len(paper_citation_paper_list_dict)))

    graph_data = {}
    graph_data[('paper', 'publishedin', 'venue')] = p_idx2v_idx_edge
    graph_data[('paper', 'writtenby', 'author')] = p_idx2a_idx_edge
    graph_data[('paper', 'citeby', 'paper')] = p_idx2p_idx_edge
    graph_data[('author', 'haspaperin', 'venue')] = a_idx2v_idx_edge
    g = dgl.heterograph(graph_data)
    return g, yearly_eid_set, p_idx2id_dict, v_name2idx_dict, a_idx2id_dict, p_id2v_name, paper_year_dict
    #######
    g = set_citation_edge_weight(g, paper_citation_paper_list_dict, p_idx2id_dict)
    g = set_mpa_norm(g, paper_citation_paper_list_dict, venue_paper_list_dict,
             venue_mapping_dict, author_paper_list_dict, a_id2idx_dict, v_name2idx_dict)

    with open(parent_path+'/dataset/a_idx2v_idx_edge.pkl', "wb" ) as pf:
        pickle.dump(a_idx2v_idx_edge, pf)
    with open(parent_path+'/dataset/apv_graph.pkl', "wb" ) as pf:
        pickle.dump(g, pf)
    with open(parent_path+'/dataset/apv_node_mapping.pkl', "wb" ) as pf:
        pickle.dump((a_id2idx_dict,p_id2idx_dict,v_name2idx_dict,a_idx2id_dict,p_idx2id_dict,v_idx2name_dict), pf)
    ########

def year_to_yid(year, interval=4):
    return (year - 1990) // interval

# If you want a yearly graph, then paper_citation_paper_list_dict should be yearly
def set_citation_edge_weight(g, paper_citation_paper_list_dict, p_idx2id_dict):
    publishedin_edge_weight = []
    p_idx2v_idx_edge = tuple(map(tuple, torch.stack(g.edges(etype='publishedin')).numpy().T))
    for (p_idx, v_idx) in p_idx2v_idx_edge:
        publishedin_edge_weight.append( len(paper_citation_paper_list_dict[ p_idx2id_dict[p_idx] ]) )

    writtenby_edge_weight = []
    p_idx2a_idx_edge = tuple(map(tuple, torch.stack(g.edges(etype='writtenby')).numpy().T))
    for (p_idx, a_idx) in p_idx2a_idx_edge:
        writtenby_edge_weight.append( len(paper_citation_paper_list_dict[ p_idx2id_dict[p_idx] ]) )

    g.edges['publishedin'].data['e_weight'] = torch.Tensor(publishedin_edge_weight)
    g.edges['writtenby'].data['e_weight'] = torch.Tensor(writtenby_edge_weight)
    return g



def set_mpa_norm(g, paper_citation_paper_list_dict,
                    venue_paper_list_dict, venue_mapping_dict, author_paper_list_dict,
                    a_id2idx_dict, v_name2idx_dict, yearly_paper_set=None):

    with open(parent_path+'/dataset/bin_paper_fos_score_vec_dict.pkl', 'rb') as handle:
        bin_paper_fos_score_vec_dict = pickle.load(handle)

    for k,v in bin_paper_fos_score_vec_dict.items():
        num_fos = bin_paper_fos_score_vec_dict[k].shape[-1]
        break
    num_venue = len(venue_mapping_dict)

    paper_v_idx_dict = {}
    for v, ps in venue_paper_list_dict.items():
        for p in ps:
            paper_v_idx_dict[p] = venue_mapping_dict[v]

    mpa = {}
    for a, ps in tqdm(author_paper_list_dict.items(), total=len(author_paper_list_dict)):
        av_matrix = np.zeros((num_venue, num_fos))
        for p in ps:
            if yearly_paper_set is not None and p not in yearly_paper_set:
                continue
            vid = paper_v_idx_dict[p]
            av_matrix[vid] += bin_paper_fos_score_vec_dict[p] * len(paper_citation_paper_list_dict[p])

        for v, _ in venue_mapping_dict.items():
            mpa[(a_id2idx_dict[a], v_name2idx_dict[v])] = sparse.csr_matrix(av_matrix[venue_mapping_dict[v]] / av_matrix.sum(axis=0, keepdims=1))

    av_edge_feat = []
    a_idx2v_idx_edge = tuple(map(tuple, torch.stack(g.edges(etype='haspaperin')).numpy().T))
    for (a,v) in a_idx2v_idx_edge:
        edge_feat = mpa[(a, v)] if (a, v) in mpa else sparse.csr_matrix(np.zeros(num_fos))
        av_edge_feat.append(edge_feat)

    g.edges['haspaperin'].data['feat'] = sparse_mx_to_torch_sparse_tensor(sparse.vstack(av_edge_feat))
    return g



# haspaperin, publishedin, writtenby
def propagate_taxo_graph():
    with open(parent_path+'/dataset/apv_graph.pkl', "rb" ) as pf:
        G = pickle.load(pf)
    with open(parent_path+'/dataset/paper_fos_score_vec_dict.pkl', 'rb') as handle:
        paper_fos_score_vec_dict = pickle.load(handle)
    papers_set = sorted(set(paper_fos_score_vec_dict.keys()))

    vec_cache = []
    for p_id in papers_set:
        vec_cache.append(paper_fos_score_vec_dict[p_id])
    num_fos = paper_fos_score_vec_dict[p_id].shape[-1]
    sp_fos_feature = sparse.vstack(vec_cache)
    sp_tensor_fos_feature = sparse_mx_to_torch_sparse_tensor(sp_fos_feature)
    embed_dict = {}
    embed_dict['paper'] = sp_tensor_fos_feature
    embed_dict['author'] = torch.zeros((G.num_nodes('author'), num_fos)).to_sparse()
    alpha = 0.5
    G.edges['publishedin'].data['e_alpha'] = torch.Tensor([1-alpha] * G.num_edges('publishedin'))
    G.edges['writtenby'].data['e_alpha'] = torch.Tensor([alpha] * G.num_edges('writtenby'))

    layer1 = PropLayer()
    layer2 = PropLayer()

    h_dict = layer1(G, embed_dict)
    h_dict = {k : F.leaky_relu(h) for k, h in h_dict.items()}
    h_dict = layer2(G, h_dict)

    print('!')




class PropLayer(nn.Module):
    def __init__(self):
        super(PropLayer, self).__init__()

    def forward(self, G, feat_dict):
        G.nodes['paper'].data['h'] = feat_dict['paper']
        G.nodes['author'].data['h'] = feat_dict['author']
        def send_a_to_v(edges):
            return {'m': edges.src['h'] * edges.data['feat'] * edges.data['e_alpha']}
        def send_p_to_v(edges):
            return {'m': edges.src['h'] * edges.data['e_alpha'] * edges.data['e_weight']}
        def send_p_to_a(edges):
            # dense_p_fos = edges.src['p_fos'].to_dense()
            # bin_p_fos = torch.zeros_like(dense_p_fos)
            # bin_p_fos[dense_p_fos > 0] = 1
            # return {'m': bin_p_fos.to_sparse() * edges.data['e_weight']}
            return {'m': edges.src['h'] * edges.data['e_weight']}

        funcs = {}
        funcs['writtenby'] = (send_p_to_a, fn.sum('m', 'h'))
        funcs['publishedin'] = (send_p_to_v, fn.sum('m', 'h'))
        funcs['haspaperin'] = (send_a_to_v, fn.sum('m', 'h'))
        G.multi_update_all(funcs, 'sum')
        return {ntype: G.nodes[ntype].data['h'] for ntype in G.ntypes}




def dgl_heterograph_testcase():
    # g = dgl.heterograph({
    #     ('user', 'follows', 'user'): [(0, 1), (1, 2)],
    #     ('user', 'plays', 'game'): [(0, 0), (1, 0), (1, 1), (2, 1)],
    #     ('developer', 'develops', 'game'): [(0, 0), (1, 1)],
    #     })

    data_dict = {
        ('user', 'follows', 'user'): (torch.tensor([0, 1]), torch.tensor([1, 2])),
        ('user', 'follows', 'topic'): (torch.tensor([1, 1]), torch.tensor([1, 2])),
        ('user', 'plays', 'game'): (torch.tensor([3, 0]), torch.tensor([1, 2]))
    }

    g = dgl.heterograph(data_dict)
    g.nodes('game') # tensor([0, 1, 2, 3, 4]), TODO: why the result is 0-4?
    g.edges(etype='plays')
    print('!')



def fos_selection():
    import numpy as np
    from sklearn.feature_selection import VarianceThreshold
    print('load paper_fos_score_vec_dict')
    with open(parent_path+'/dataset/paper_fos_score_vec_dict.pkl', 'rb') as handle:
        paper_fos_score_vec_dict = pickle.load(handle)

    vec_cache = []
    p_list = []
    for k,v in paper_fos_score_vec_dict.items():
        p_list.append(k)
        vec_cache.append(v.toarray())
    vec_cache = np.array(vec_cache)
    vec_cache = vec_cache.reshape((vec_cache.shape[0], vec_cache.shape[-1]))
    sel = VarianceThreshold(threshold=(.95 * (1 - .95)))
    selected_f = sel.fit_transform(vec_cache)

    # TODO: bin_paper_fos_score_vec_dict also need feat selection with same mask (mask = sel.get_support()), then save to filtered_bin_paper_fos_score_vec_dict
    #  bin_paper_fos_score_vec_dict was loaded in  function: set_mpa_norm.
    #  filtered_bin_paper_fos_score_vec_dict and filtered_fos_vec_dict should be used in the same time.

    filtered_paper_fos_score_vec_dict = {}
    for i in range(selected_f.shape[0]):
        filtered_paper_fos_score_vec_dict[p_list[i]] = selected_f[i]

    print('{} -> {}'.format(vec_cache.shape[1], filtered_paper_fos_score_vec_dict.shape[1]))

    with open(parent_path+'/dataset/filtered_fos_vec_dict.pkl', 'wb') as handle:
        pickle.dump(filtered_paper_fos_score_vec_dict, handle)


if __name__ == '__main__':
    construct_hetergraph()
    # dgl_heterograph_testcase()
    # fos_selection()
   # propagate_taxo_graph()