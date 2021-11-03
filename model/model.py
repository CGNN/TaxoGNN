import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import itertools
import numpy as np
import scipy.sparse as sp
from IPython import embed
import torch.utils.data as tdata
from tqdm import tqdm

import dgl
import time
# from dgl.nn import SAGEConv
import dgl.nn as dglnn
import dgl.function as fn
from dgl.utils import expand_as_pair, DGLError

# from layer import HeteroGraphConv
from dgl.nn import HeteroGraphConv
import math

BERT_VEC_LENS = 768

class TaxoHeteroGraphConv(HeteroGraphConv):
    def __init__(self, mods, aggregate='sum'):
        super(TaxoHeteroGraphConv, self).__init__(mods, aggregate)

    def forward(self, g, inputs, mod_args=None, mod_kwargs=None):
        """Forward computation
        Invoke the forward function with each module and aggregate their results.
        Parameters
        ----------
        g : DGLHeteroGraph
            Graph data.
        inputs : dict[str, Tensor] or pair of dict[str, Tensor]
            Input node features.
        mod_args : dict[str, tuple[any]], optional
            Extra positional arguments for the sub-modules.
        mod_kwargs : dict[str, dict[str, any]], optional
            Extra key-word arguments for the sub-modules.
        Returns
        -------
        dict[str, Tensor]
            Output representations for every types of nodes.
        """
        if mod_args is None:
            mod_args = {}
        if mod_kwargs is None:
            mod_kwargs = {}
        outputs = {nty: [] for nty in g.dsttypes}
        if isinstance(inputs, tuple) or g.is_block:
            if isinstance(inputs, tuple):
                src_inputs, dst_inputs = inputs
            else:
                src_inputs = inputs
                # below is used to do graphsage like function
                # dst_inputs = {k: v[:g.number_of_dst_nodes(k)] for k, v in inputs.items()}

            for etype in self.mods.keys():
                # for stype, etype, dtype in g.canonical_etypes:
                stype, etype, dtype = g.to_canonical_etype(etype)
                rel_graph = g[stype, etype, dtype]
                if rel_graph.number_of_edges() == 0:
                    continue
                # if stype not in src_inputs or dtype not in dst_inputs:
                if stype not in src_inputs:
                    continue

                dstdata = self.mods[etype](
                    rel_graph,
                    (src_inputs[stype], src_inputs[dtype]),
                    *mod_args.get(etype, ()),
                    **mod_kwargs.get(etype, {}))
                outputs[dtype].append(dstdata)
        else:
            for etype in self.mods.keys():
                # for stype, etype, dtype in g.canonical_etypes:
                stype, etype, dtype = g.to_canonical_etype(etype)
                rel_graph = g[stype, etype, dtype]
                if rel_graph.number_of_edges() == 0:
                    continue
                if stype not in inputs:
                    continue
                print(stype, etype, dtype)
                # embed()
                dstdata = self.mods[etype](
                    rel_graph,
                    (inputs[stype], inputs[dtype]),
                    *mod_args.get(etype, ()),
                    **mod_kwargs.get(etype, {}))
                outputs[dtype].append(dstdata)
        rsts = {}
        for nty, alist in outputs.items():
            if len(alist) != 0:
                rsts[nty] = self.agg_fn(alist, nty)
        return rsts


class SimpleGraphConv(dglnn.GraphConv):

    def __init__(self,
                 in_feats,
                 out_feats,
                 edge_feature_dim=False,
                 norm='both',
                 weight=True,
                 bias=True,
                 activation=None,
                 allow_zero_in_degree=False,
                 attn=False):
        super(SimpleGraphConv, self).__init__(in_feats, out_feats, 'none', weight, bias, activation,
                                              allow_zero_in_degree)
        self._norm = norm
        # if attn is not None:
        #    self.attn = True
        # else:
        self.attn = attn
        self.reset_parameters()

    def edge_udf(self, edges):
        if 'h' in edges.src and 'w' in edges.data:
            # m indicates
            # print((edges.src['w'].unsqueeze(1) * edges.data['w']).shape)
            # multi_w = edges.src['w'].unsqueeze(1) * edges.data['w']
            # print(edges.src['h'].shape, edges.data['w'].shape)
            # return {'h' : torch.einsum('bi,bj->bij', edges.src['h'], edges.data['w'][:,0])}
            return {'h': edges.src['h'] * edges.data['w'][:, self.k].unsqueeze(1)}
        elif 'h' in edges.src and 'w' in edges.src:
            return {'h': edges.src['h'], 'w': edges.src['w']}
        elif 'w' in edges.src:
            # w indicates citation always
            return {'w': edges.src['w']}
        else:
            return {'h': edges.src['h']}

    def sum_udf(self, nodes):
        # print(nodes.mailbox['h'].shape)
        return {'h': torch.sum(nodes.mailbox['h'], dim=1)}
        # return {'h':torch.stack([torch.sum(nodes.mailbox[k], dim=1) for k in range(10)], dim=2)}

    def mean_udf(self, nodes):
        # embed()
        # [#nodeinbatch, in_degree, dim]
        # if self.dropout is not None and self.training:
        if self.attn == True:
            n = torch.zeros((nodes.mailbox['h'].shape[0], 10, nodes.mailbox['h'].shape[2]),
                            device=nodes.mailbox['h'].device)
            n[:, :min(nodes.mailbox['h'].shape[1], 10), :] = nodes.mailbox['h'][:,
                                                             :min(nodes.mailbox['h'].shape[1], 10), :]

            if self.training and nodes.mailbox['h'].shape[1] > 1:
                idx = torch.bernoulli(0.5 * torch.ones(10)) == 0
                if idx[:nodes.mailbox['h'].shape[1]].sum() == nodes.mailbox['h'].shape[1]:
                    n[:, 1:, :] = 0
                else:
                    n[:, idx, :] = 0
                if n.sum() == 0:
                    embed()
            #    return {'h': nodes.mailbox['h'][:,0,:]}
            # else:
            return {'h': n}
        else:
            return {'h': torch.mean(nodes.mailbox['h'].float(), dim=1)}
            # return {'h': torch.mean(nodes.mailbox['h'].float(), dim=1)}
        # top_k = 0.1
        # print(nodes.mailbox['m'].shape, nodes.mailbox['w'].shape)
        # else:
        #    top_k_idx = nodes.mailbox['w'].argsort(dim=1, descending=True).view(-1)[:math.ceil(self.top_k * nodes.mailbox['w'].shape[1])]
        # return {'h': th.mean(nodes.mailbox['h'][:, top_k_idx].float(), dim=1)}
        #    return {'h': torch.mean(nodes.mailbox['h'][:, top_k_idx].float(), dim=1)}
        # return {'h': th.sum(torch.ones_like(nodes.mailbox['w']), dim=1)}
        # return {'h': th.sum(nodes.mailbox['w'], dim=1)}

    def forward(self, graph, feat, weight=None, edge_weight=None, degree_scalar=None):
        r"""
        Description
        -----------
        Compute graph convolution.
        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, it represents the input feature of shape
            :math:`(N, D_{in})`
            where :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, which is the case for bipartite graph, the pair
            must contain two tensors of shape :math:`(N_{in}, D_{in_{src}})` and
            :math:`(N_{out}, D_{in_{dst}})`.
        weight : torch.Tensor, optional
            Optional external weight tensor.
        Returns
        -------
        torch.Tensor
            The output feature
        Raises
        ------
        DGLError
            Case 1:
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.
            Case 2:
            External weight is provided while at the same time the module
            has defined its own weight parameter.
        Note
        ----
        * Input shape: :math:`(N, *, \text{in_feats})` where * means any number of additional
          dimensions, :math:`N` is the number of nodes.
        * Output shape: :math:`(N, *, \text{out_feats})` where all but the last dimension are
          the same shape as the input.
        * Weight shape: :math:`(\text{in_feats}, \text{out_feats})`.
        """
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')
            aggregate_fn = fn.copy_src('h', 'm')
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.number_of_edges()
                graph.edata['_edge_weight'] = edge_weight
                aggregate_fn = fn.u_mul_e('h', '_edge_weight', 'm')

            # (BarclayII) For RGCN on heterogeneous graphs we need to support GCN on bipartite.
            feat_src, feat_dst = expand_as_pair(feat, graph)
            if self._norm == 'both':
                degs = graph.out_degrees().float().clamp(min=1)
                norm = torch.pow(degs, -0.5)
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = torch.reshape(norm, shp)
                feat_src = feat_src * norm

            if weight is not None:
                if self.weight is not None:
                    raise DGLError('External weight is provided while at the same time the'
                                   ' module has defined its own weight parameter. Please'
                                   ' create the module with flag weight=False.')
            else:
                weight = self.weight

            if self._in_feats > self._out_feats:
                # if False:
                # mult W first to reduce the feature size for aggregation.
                if weight is not None:
                    feat_src = torch.matmul(feat_src, weight)
                graph.srcdata['h'] = feat_src
                if self._norm == 'sum':
                    graph.update_all(self.edge_udf, self.sum_udf)
                elif self._norm == 'mean':
                    graph.update_all(self.edge_udf, self.mean_udf)
                rst = graph.dstdata['h']
                if degree_scalar is not None and self._norm == 'mean':
                    if '_ID' in graph.dstdata:
                        rst = (degree_scalar[graph.dstdata['_ID']].unsqueeze(1) + 1).log() / (
                                    degree_scalar + 1).log().mean() * rst
                    else:
                        # embed()
                        rst = (degree_scalar + 1).log().unsqueeze(1) / (degree_scalar + 1).log().mean() * rst

            else:
                # aggregate first then mult W
                if 'w' in graph.srcdata:
                    # print(feat_src.shape, graph.srcdata['fos'].shape)
                    graph.srcdata['h'] = feat_src
                    graph.srcdata['w'] = graph.srcdata['w']
                else:
                    graph.srcdata['h'] = feat_src
                if self._norm == 'sum':
                    graph.update_all(self.edge_udf, self.sum_udf)
                elif self._norm == 'mean':
                    graph.update_all(self.edge_udf, self.mean_udf)
                # change it back
                if degree_scalar is not None:
                    if '_ID' in graph.dstdata:
                        rst = (degree_scalar[graph.dstdata['_ID']].unsqueeze(1) + 1).log() / (
                                    degree_scalar + 1).log().mean() * graph.dstdata['h']
                    else:
                        # embed()
                        rst = (degree_scalar + 1).log().unsqueeze(1) / (degree_scalar + 1).log().mean() * graph.dstdata[
                            'h']
                else:
                    rst = graph.dstdata['h']
                if weight is not None:
                    try:
                        rst = torch.matmul(rst, weight)
                    except:
                        embed()
                    # rst = weight(rst)
            '''
            if self._norm != 'none':
                degs = graph.in_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = th.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = th.reshape(norm, shp)
                rst = rst * norm
            '''
            if self.bias is not None:
                rst = rst + self.bias

            if self._activation is not None:
                rst = self._activation(rst)

            return rst


class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats=None, rel_names=[], self_loop=None):
        super().__init__()
        # call update every time-frame \Delta, A^t
        conv1_funcs = {
            'writtenby': SimpleGraphConv(in_feats, hid_feats, bias=None, norm='sum', weight=False,
                                         allow_zero_in_degree=True),
            'citeby': SimpleGraphConv(in_feats, hid_feats, bias=None, weight=False, norm='sum',
                                      allow_zero_in_degree=True),
            # 'publishedin': SimpleGraphConv(in_feats, hid_feats, bias=None,weight=False, norm='sum', allow_zero_in_degree=True),
            # 'haspaperin': None
        }

        #
        conv2_funcs = {
            'publishedin': SimpleGraphConv(in_feats, hid_feats, bias=None, norm='mean', weight=False,
                                           allow_zero_in_degree=True, top_k=0.1),
            'haspaperin': SimpleGraphConv(in_feats, hid_feats, bias=None, norm='sum', weight=False,
                                          allow_zero_in_degree=True),
        }

        self.conv1 = TaxoHeteroGraphConv(conv1_funcs, aggregate='sum')
        print(self.conv1)
        self.conv2 = TaxoHeteroGraphConv(conv2_funcs, aggregate='sum')
        print(self.conv2)

    def forward(self, graph, inputs, k):
        # inputs are features of nodes
        # print(inputs)
        h = self.conv1(graph, inputs, k)
        # print(h)
        # h = {k: F.relu(v) for k, v in h.items()}
        # h = {k: F.relu(h[k]) for k in h}
        # h = self.conv2(graph, h)
        return h

    def prop_venue(self, graph, inputs):
        h = self.conv2(graph, inputs)
        return h

    def prop_author(self, graph, inputs):
        h = self.conv1(graph, inputs)
        return h


class BatchRGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, use_taxon, attn=False):
        super().__init__()
        # print(rel_names)
        # embed()
        self.attn = attn
        conv1_funcs = {
            'paper_selfloop': SimpleGraphConv(BERT_VEC_LENS + in_feats if use_taxon else BERT_VEC_LENS, out_feats, bias=None, norm='sum', weight=True,
                                              allow_zero_in_degree=True),
            'author_selfloop': SimpleGraphConv(in_feats, hid_feats, bias=None, norm='sum', weight=True,
                                               allow_zero_in_degree=True),
            'writtenby': SimpleGraphConv(BERT_VEC_LENS + in_feats if use_taxon else BERT_VEC_LENS, hid_feats, bias=None, norm='mean', weight=True,
                                         allow_zero_in_degree=True),
            # 'citeby': SimpleGraphConv(in_feats, hid_feats, bias=None,weight=True, norm='mean', allow_zero_in_degree=True),
            'publishedin': SimpleGraphConv(BERT_VEC_LENS + in_feats if use_taxon else BERT_VEC_LENS, hid_feats, bias=None, weight=True, norm='mean',
                                           allow_zero_in_degree=True),
        }

        self.conv1 = TaxoHeteroGraphConv(conv1_funcs, aggregate='stack')

        conv2_funcs = {
            'paper_selfloop': SimpleGraphConv(hid_feats, out_feats, bias=None, norm='mean', weight=False,
                                              allow_zero_in_degree=True, attn=self.attn),
            # 'citeby': SimpleGraphConv(hid_feats, out_feats, bias=None, norm='mean', weight=True, allow_zero_in_degree=True),
            'publishedin_reverse': SimpleGraphConv(hid_feats, out_feats, bias=None, weight=True, norm='mean',
                                                   allow_zero_in_degree=True, attn=self.attn),
            'writtenby_reverse': SimpleGraphConv(2 * hid_feats, out_feats, bias=None, norm='mean', weight=True,
                                                 allow_zero_in_degree=True, attn=self.attn),
            # rel: SimpleGraphConv(hid_feats, out_feats, bias=None, norm='mean', weight=True, allow_zero_in_degree=True)
            # for rel in rel_names
        }

        self.conv2 = TaxoHeteroGraphConv(conv2_funcs, aggregate='sum')

    def forward(self, blocks, inputs, mod_kwargs):
        # inputs are features of nodes
        h = self.conv1(blocks[0], inputs, mod_kwargs=mod_kwargs)
        for k in h:
            h[k] = h[k].reshape(h[k].shape[0], -1)

        h = {k: F.normalize(F.silu(v), p=2, dim=1) for k, v in h.items()}
        # h = {k: F.silu(v) for k, v in h.items()}
        # return h
        # h = {k: F.relu(h[k]) for k in h}

        h = self.conv2(blocks[1], h)

        if not self.attn:
            for k in h:
                h[k] = h[k].reshape(h[k].shape[0], -1)

        return h

    def inference(self, graph, inputs, mod_kwargs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs, mod_kwargs=mod_kwargs)
        for k in h:
            h[k] = h[k].reshape(h[k].shape[0], -1)

        intermediate_h = {k: F.normalize(F.silu(v), p=2, dim=1) for k, v in h.items()}
        # intermediate_h = {k: F.silu(v) for k, v in h.items()}

        # return h
        # h = {k: F.relu(h[k]) for k in h}
        h = self.conv2(graph, intermediate_h)
        if not self.attn:
            for k in h:
                h[k] = h[k].reshape(h[k].shape[0], -1)
        # embed()
        return h, intermediate_h


class BatchHeteroDotProductPredictor(nn.Module):
    def __init__(self, n_hidden, etype='citeby'):
        super().__init__()
        # BERT_VEC_LENS = 768
        # self.lin1 = nn.Linear(3 * n_hidden + BERT_VEC_LENS, 128)
        # self.lin1 = nn.Linear(3 * n_hidden, 3 * n_hidden)
        # self.lin2 = nn.Linear(3 * n_hidden, 3 * n_hidden)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.etype = etype
        # self.lm_emb = bert_embedding

    def cos_udf(self, edges):
        return {'score': -self.cos(edges.src['h'], edges.dst['h'])}

    def cross_attn(self, edges):

        a, b = edges.src['h'][:, 2, :, :], edges.dst['h'][:, 2, :, :]
        attn_weights = torch.exp(-torch.norm(a[:, :, None] - b[:, None], dim=3, p=2))
        mask_a = a.sum(dim=2) != 0
        # attn_weights[mask_a,:]=0
        mask_b = b.sum(dim=2) != 0
        attn_weights = attn_weights * mask_a[:, :, None].float() * mask_b[:, None].float()
        a_weight = attn_weights.sum(dim=2) / attn_weights.sum(dim=2).sum(dim=1)[:, None]
        b_weight = attn_weights.sum(dim=1) / attn_weights.sum(dim=2).sum(dim=1)[:, None]
        final_a = (a * a_weight[:, :, None]).sum(dim=1)
        final_b = (b * b_weight[:, :, None]).sum(dim=1)
        # final_src = torch.cat([edges.src['h'][:,0,:,:].sum(dim=2), edges.src['h'][:,1,:,:].sum(dim=2),final_a], dim=1)
        # embed()
        final_src = edges.src['h'][:, 0, :, :].sum(dim=1) + edges.src['h'][:, 1, :, :].sum(dim=1) + final_a
        final_dst = edges.dst['h'][:, 0, :, :].sum(dim=1) + edges.dst['h'][:, 1, :, :].sum(dim=1) + final_b
        return {'score': final_src - final_dst}
        embed()

    def forward(self, graph, h, attn=False, cosine=False):
        # h contains the node representations for each edge type computed from
        # the GNN for heterogeneous graphs defined in the node classification
        # section (Section 5.1).

        with graph.local_scope():
            graph.ndata['h'] = h
            etype = self.etype
            if graph.number_of_edges(etype) == 0:
                return
            if attn:
                graph.apply_edges(self.cross_attn, etype=etype)
                return graph.edges[etype].data['score'].norm(p=2, dim=1)
                #return graph.edges[etype].data['score']
            elif cosine:
                graph.apply_edges(self.cos_udf, etype=etype)
                return graph.edges[etype].data['score']
            else:
                graph.apply_edges(fn.u_sub_v('h', 'h', 'score'), etype=etype)
                return graph.edges[etype].data['score'].norm(p=2, dim=1)


class BatchModel(nn.Module):
    def __init__(self, node_embedding_size, hidden_features, out_features, use_taxon, num_node_dict, num_taxons,
                 init_embedding=None, bert_embedding=None, attn=False, device='cpu'):
        super().__init__()
        # self.node_embeddings = {nt:nn.Embedding(num_node_dict[nt], node_embedding_size) for nt in num_node_dict}
        # self.node_embeddings = {nt:Parameter(torch.FloatTensor(num_node_dict[nt], node_embedding_size)) for nt in num_node_dict}
        self.node_embeddings = {}
        for k in num_taxons:
            if init_embedding[k] is not None:
                self.node_embeddings[k] = nn.Embedding.from_pretrained(init_embedding[k], padding_idx=num_taxons[k]).to(
                    device)
            else:
                self.node_embeddings[k] = nn.Embedding(num_taxons[k] + 1, node_embedding_size,
                                                       padding_idx=num_taxons[k]).to(device)
            # self.node_embeddings = torch.FloatTensor(num_taxons+1, node_embedding_size).cuda()
        BERT_VEC_LENS = 768
        # self.lin1 = nn.Linear(3 * out_features + BERT_VEC_LENS, 3 * out_features)
        # self.lin2 = nn.Linear(3 * out_features, 3 * out_features)

        # self.node_embeddings['author'].weight.requires_grad = False

        # self.node_embeddings.uniform_(-1.0, 1.0)
        # self.node_embeddings[num_taxons,:] = 0
        self.use_taxon = use_taxon
        self.num_taxons = num_taxons
        self.attn = attn
        self.rgcn = BatchRGCN(node_embedding_size, hidden_features, out_features,use_taxon, attn)
        self.pred = BatchHeteroDotProductPredictor(out_features)
        self.bert_embedding = bert_embedding

        # self._lambda = 1e-5
        self._lambda = 0.0

        # self.reset_parameters(init_embedding)

    def reset_parameters(self, init_embedding=None):
        if init_embedding is None:
            for i, nt in enumerate(self.nt_list):
                self.node_embeddings[i].data.uniform_(-1.0, 1.0)
        else:
            for i, nt in enumerate(self.nt_list):
                self.node_embeddings[i].data = init_embedding[nt]

    def forward(self, pos_g, neg_g, blocks, inputs, input_weights, degree_scalar):
        # x = {nt:self.node_embeddings[i][blocks[0].srcdata['_ID'][nt]] for i,nt in enumerate(self.nt_list) }
        x = {}

        if self.bert_embedding is not None:
            lm_emb = self.bert_embedding[pos_g.ndata['_ID']['paper'], :]
        else:
            lm_emb = None
        for nt in inputs:
            if inputs[nt] is not None:
                if nt == 'paper':
                    if self.use_taxon:
                    # x[nt] = self.bert_embedding[blocks[0].srcdata['_ID']['paper'], :]
                    # embed()
                        x[nt] = torch.cat(
                            [(self.node_embeddings[nt](inputs[nt][blocks[0].srcdata['_ID'][nt], :].long())).sum(dim=1),
                            self.bert_embedding[blocks[0].srcdata['_ID']['paper'], :]], dim=1)
                    else:
                        x[nt] = self.bert_embedding[blocks[0].srcdata['_ID']['paper'], :]
                else:
                    x[nt] = (self.node_embeddings[nt](inputs[nt][blocks[0].srcdata['_ID'][nt], :].long()) *
                             input_weights[nt][blocks[0].srcdata['_ID'][nt], :].unsqueeze(2)).sum(dim=1)
                # x[nt] = (self.node_embeddings[nt](inputs[nt][blocks[0].srcdata['_ID'][nt], :].long()) ).sum(dim=1)
                # x[nt] = (self.node_embeddings[inputs[nt][blocks[0].srcdata['_ID'][nt], :].long()] * input_weights[nt][blocks[0].srcdata['_ID'][nt], :].unsqueeze(2)).sum(dim=1)
                # x[nt] = (self.node_embeddings(inputs[nt][blocks[0].srcdata['_ID'][nt], :].long()) ).sum(dim=1)
            else:
                x[nt] = None
        # embed()
        h = self.rgcn(blocks, x, mod_kwargs=degree_scalar)
        # embed()
        # h['paper'] = self.lin2(F.relu(self.lin1(h['paper'])))
        # h['paper'] = self.lin1(F.relu(h['paper']))

        # if lm_emb is not None:
        #    h['paper'] = self.lin2(F.silu(self.lin1( torch.cat([h['paper'], lm_emb], dim=1))))

        if self._lambda > 0:
            # reg_loss = self._lambda * 1/2 * torch.norm(self.node_embeddings(self.reg_edges[:, 0]) - self.node_embeddings(self.reg_edges[:, 1]), p=2) ** 2
            neg_samples = torch.randint(0, self.num_taxons, (self.reg_edges.shape[0], 10)).cuda()
            pos_pairs = torch.norm(self.node_embeddings['paper'](self.reg_edges[:, 0]) - self.node_embeddings['paper'](
                self.reg_edges[:, 1]), p=2, dim=1)
            neg_pairs = torch.norm(
                self.node_embeddings['paper'](self.reg_edges[:, 0]).unsqueeze(1) - self.node_embeddings['paper'](
                    neg_samples), p=2, dim=2)
            prob = torch.exp(-pos_pairs) / (torch.exp(-neg_pairs).sum(dim=1) + torch.exp(-pos_pairs))
            reg_loss = self._lambda * prob.log().sum()
            # embed()
        else:
            reg_loss = 0.0
        return self.pred(pos_g, h, self.attn), self.pred(neg_g, h, self.attn), reg_loss

    def node_inference(self, input_nodes, blocks, inputs, input_weights, degree_scalar):
        x = {}
        if self.bert_embedding is not None:
            lm_emb = self.bert_embedding[input_nodes['paper'], :]
        else:
            lm_emb = None
        for nt in inputs:
            if inputs[nt] is not None:
                if nt == 'paper':
                    # x[nt] = self.bert_embedding[blocks[0].srcdata['_ID']['paper'], :]
                    # embed()
                    if self.use_taxon:
                        x[nt] = torch.cat(
                            [(self.node_embeddings[nt](inputs[nt][blocks[0].srcdata['_ID'][nt], :].long())).sum(dim=1),
                            self.bert_embedding[blocks[0].srcdata['_ID']['paper'], :]], dim=1)
                    else:
                        x[nt] = self.bert_embedding[blocks[0].srcdata['_ID']['paper'], :]
                else:
                    x[nt] = (self.node_embeddings[nt](inputs[nt][blocks[0].srcdata['_ID'][nt], :].long()) *
                             input_weights[nt][blocks[0].srcdata['_ID'][nt], :].unsqueeze(2)).sum(dim=1)
                    # x[nt] = (self.node_embeddings[nt](inputs[nt][blocks[0].srcdata['_ID'][nt], :].long()) ).sum(dim=1)
                # x[nt] = (self.node_embeddings[inputs[nt][blocks[0].srcdata['_ID'][nt], :].long()] * input_weights[nt][blocks[0].srcdata['_ID'][nt], :].unsqueeze(2)).sum(dim=1)
                # x[nt] = (self.node_embeddings(inputs[nt][blocks[0].srcdata['_ID'][nt], :].long()) ).sum(dim=1)
            else:
                x[nt] = None
        # embed()
        h = self.rgcn(blocks, x, mod_kwargs=degree_scalar)
        return h

    def predict(self, g, pos_g, neg_g, inputs, input_weights, degree_scalar):
        x = {}
        for nt in inputs:
            if inputs[nt] is not None:
                if nt == 'paper':
                    # x[nt] = self.bert_embedding
                    # x[nt] = (self.node_embeddings[nt](inputs[nt].long()) * input_weights[nt].unsqueeze(2)).sum(dim=1)
                    # x[nt] = (self.node_embeddings[inputs[nt].long()] * input_weights[nt].unsqueeze(2)).sum(dim=1)
                    if self.use_taxon:
                        x[nt] = torch.cat([(self.node_embeddings[nt](inputs[nt].long())).sum(dim=1), self.bert_embedding],
                                        dim=1)
                    else:
                        x[nt] = self.bert_embedding
                else:
                    x[nt] = (self.node_embeddings[nt](inputs[nt].long()) * input_weights[nt].unsqueeze(2)).sum(dim=1)
            else:
                x[nt] = None
        # embed()
        h, inter_h = self.rgcn.inference(g, x, mod_kwargs=degree_scalar)
        # if self.bert_embedding is not None:
        #    h['paper'] = self.lin2(F.silu(self.lin1( torch.cat([h['paper'], self.bert_embedding], dim=1))))
        # h['paper'] = self.lin2(F.relu(self.lin1(h['paper'])))
        # h['paper'] = self.lin1(F.relu(h['paper']))
        return self.pred(pos_g, h, self.attn), self.pred(neg_g, h, self.attn), h, inter_h