from torch_clustering import PyTorchKMeans
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import heapq
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

from .gae_simple import IGAE_encoder, GNNLayer
from .gcn import IGAE_decoder

# from layers.mlfpn_gcn import MLFPN_GCN
from layers.cluster_layer import ClusteringHead

from itertools import combinations
from collections import defaultdict







class SelfExpressiveLayer(nn.Module):
    def __init__(self, num_samples, init_affinity=None, device='cpu'):
        super(SelfExpressiveLayer, self).__init__()
        self.init_affinity = init_affinity
        self.device = device
        self.affinity_mat = nn.Parameter(torch.randn(num_samples, num_samples), requires_grad=True).to(device)
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_affinity is not None:
            if not isinstance(self.init_affinity, torch.Tensor):
                self.init_affinity = torch.from_numpy(self.init_affinity).float()
            self.affinity_mat.data = self.init_affinity
        # else:
        #     nn.init.constant_(self.affinity_mat, 1.)

    def forward(self, x):
        latent_recon = torch.matmul(self.affinity_mat, x)
        return latent_recon


class EdgeDecoder(nn.Module):
    """Simple MLP Edge Decoder"""

    def __init__(
            self, in_channels, hidden_channels, out_channels=1,
            num_layers=1, dropout=0.5, activation='relu'
    ):

        super().__init__()
        self.mlps = nn.ModuleList()

        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.mlps.append(nn.Linear(first_channels, second_channels))

        self.dropout = nn.Dropout(dropout)
        self.activation = creat_activation_layer(activation)

    def forward(self, z_1, z_2, edge, sigmoid=True, reduction=False):
        x = z_1[edge[0]] * z_2[edge[1]]

        if reduction:
            x = x.mean(1)

        for i, mlp in enumerate(self.mlps[:-1]):
            x = self.dropout(x)
            x = mlp(x)
            x = self.activation(x)
        x = self.mlps[-1](x)

        if sigmoid:
            return x.sigmoid()
        else:
            return x


def creat_activation_layer(activation):
    if activation is None:
        return nn.Identity()
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "elu":
        return nn.ELU()
    elif activation == "prelu":
        return nn.PReLU()
    else:
        raise ValueError("Unknown activation")


class MLP_AG(nn.Module):
    def __init__(self, n_mlp):
        super(MLP_AG, self).__init__()
        self.w = nn.Linear(n_mlp, 2)

    def forward(self, mlp_in):
        weight_output1 = F.softmax(F.leaky_relu(self.w(mlp_in)), dim=1)
        return weight_output1


class SingleViewModel(nn.Module):
    def __init__(self, in_channel, gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, gae_n_dec_1, gae_n_dec_2, gae_n_dec_3,
                 n_samples, n_clusters, use_se, topk, if_mask, device):
        super(SingleViewModel, self).__init__()
        self.use_se = use_se
        self.if_mask = if_mask
        self.mask_predictor = nn.Linear(gae_n_enc_3, 1)
        self.mask_token = nn.Parameter(torch.zeros(1, in_channel))
        self.mask_mode = 'near_first'  # 'near_first' or 'far_first'
        self.encoder = IGAE_encoder(gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, in_channel)
        self.decoder = IGAE_decoder(gae_n_dec_1, gae_n_dec_2, gae_n_dec_3, in_channel)
        self.self_expr_layer = SelfExpressiveLayer(n_samples, init_affinity=None, device=device)

        self.gnn_4 = GNNLayer(gae_n_enc_3, n_clusters)

        self.cluster_layer = ClusteringHead(n_clusters, gae_n_enc_3)
        self.mlp = nn.Sequential(
            nn.Linear(gae_n_enc_3, gae_n_enc_3),
            nn.BatchNorm1d(gae_n_enc_3),
            nn.Sigmoid(),
            nn.Dropout(0.3),
        )
        # self.edge_redefine = EdgeRedefine(n_samples, topk)
        self.agcn_ag = MLP_AG(2 * n_samples)

    def forward(self, x, adj, epoch=0, T=300, init_rate=0.5):
        # num_nodes = x.shape[0]
        # # 随机选择node_mask_rate * n个样本
        # perm = torch.randperm(num_nodes, device=x.device)
        # # 计算需要掩码的节点数量
        # num_mask_nodes = int(node_mask_rate * num_nodes)
        # mask_nodes = perm[: num_mask_nodes]
        # keep_nodes = perm[num_mask_nodes: ]\
        # ---------- Step 1: get initial feature & cluster ----------
        feat_pre, _ = self.encoder(x, adj)
        q_full = self.cluster_layer(feat_pre).detach()

        if self.training and self.if_mask:
            labels = q_full.argmax(1)
            cluster_prob = torch.zeros_like(labels, dtype=x.dtype)
            for label in labels.unique():
                members = labels == label
                values = feat_pre[members].detach()
                distance = (values - values.mean(0)).square().sum(1)
                cluster_prob[members] = (-distance / distance.mean().clamp_min(1e-8)).softmax(0)
            weight = 0.1 + 0.8 * epoch / T
            graph_prob = self.mask_predictor(feat_pre).flatten().sigmoid()
            prob = ((1 - weight) * cluster_prob + weight * graph_prob).clamp(1e-6, 1 - 1e-6)
            logits = torch.stack((torch.log1p(-prob), prob.log()), dim=1)
            mask = F.gumbel_softmax(logits, tau=1.0, hard=True)[:, 1:2]
            self.masked_nodes = mask.detach().flatten().bool()
            x_node1 = x + mask * (self.mask_token - x)
        else:
            x_node1 = x

        self.edge_inputs = x_node1
        feat1, z_igae_adj = self.encoder(x_node1, adj)
        # feat2, z_igae_adj = self.encoder(x_node2, adj)
        if self.use_se:
            latent_recon1 = self.self_expr_layer(feat1)
            # latent_recon2 = self.self_expr_layer(feat2)
            # latent_drop = self.mlp(latent_recon)
            recon1, z_hat_adj = self.decoder(latent_recon1, adj)
            # recon2, z_hat_adj  = self.decoder(latent_recon2, adj)
        else:
            latent_recon1 = None
            # latent_recon2 = None
            # latent_drop = self.mlp(feat)
            recon1, z_hat_adj = self.decoder(feat1, adj)
            # recon2, z_hat_adj = self.decoder(feat2, adj)
        # feat2 = feat1
        # recon2 = recon1
        # latent_recon2 = latent_recon1

        # feat = (feat1 + feat2) / 2
        # recon = (recon1 + recon2) / 2
        # latent_recon = (latent_recon1 + latent_recon2) / 2

        feat = feat1
        feat1 = self.gnn_4(feat_pre, adj, active=False)
        recon = recon1
        latent_recon = latent_recon1

        q = self.cluster_layer(feat)

        return feat, recon, latent_recon, z_hat_adj, q, feat1, q_full

    # def forward(self, x, adj, node_mask_rate=0.5):
    #     num_nodes = x.shape[0]
    #     # 随机选择node_mask_rate * n个样本
    #     perm = torch.randperm(num_nodes, device=x.device)
    #     # 计算需要掩码的节点数量
    #     num_mask_nodes = int(node_mask_rate * num_nodes)
    #     mask_nodes = perm[: num_mask_nodes]
    #     keep_nodes = perm[num_mask_nodes: ]

    #     # 复制x_node
    #     x_node1 = x.clone()
    #     x_node2 = x.clone()
    #     x_node1[mask_nodes] = 0.0
    #     x_node2[keep_nodes] = 0.0

    #     feat, z_igae_adj = self.encoder(x_node1, adj)
    #     feat, z_igae_adj = self.encoder(x_node1, adj)
    #     if self.use_se:
    #         latent_recon = self.self_expr_layer(feat)
    #         # latent_drop = self.mlp(latent_recon)
    #         recon, z_hat_adj  = self.decoder(latent_recon, adj)
    #     else:
    #         latent_recon = None
    #         # latent_drop = self.mlp(feat)
    #         recon, z_hat_adj = self.decoder(feat, adj)
    #     q = self.cluster_layer(feat)
    #     return feat, recon, latent_recon, z_hat_adj, q

    def build_knn_graph(self, x, k=50):
        from sklearn.neighbors import NearestNeighbors
        x_cpu = x.detach().cpu().numpy()
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(x_cpu)
        _, indices = nbrs.kneighbors(x_cpu)

        # 批量构建边对，跳过自身
        rows = np.repeat(np.arange(len(indices)), k)
        cols = indices[:, 1:].reshape(-1)
        edges = np.stack([rows, cols], axis=1)

        return set(map(tuple, edges))

    def get_cluster_edges(self, x, adj):
        self.eval()
        with torch.no_grad():
            feat, _ = self.encoder(x, adj)
            q = self.cluster_layer(feat)
            cluster_ids = torch.argmax(q, dim=1).cpu().numpy()

        # 将节点按 cluster_id 分组
        cluster_groups = defaultdict(list)
        for idx, cid in enumerate(cluster_ids):
            cluster_groups[cid].append(idx)

        # 同一簇内所有两两组合
        cluster_edges = set()
        for group in cluster_groups.values():
            if len(group) < 2:
                continue
            cluster_edges.update(combinations(group, 2))  # i<j 自动满足

        return cluster_edges

    def forward_edge_refine(self, x, adj, x_z, apply_edge_mask=False, masked_edges=None, knn_k=50):
        ## [begin] 边重定义
        # x_array = list(np.shape(x))
        # n_x = x_array[0]
        #
        # if x.equal(x_z):
        #     adj = adj
        # else:
        #     # V1
        #     adj1_tensor = adj
        #     z4adj2 = x_z.data.cpu().numpy()
        #     adj2 = adj_knn1(z4adj2)
        #     # adj2 = adj2.cuda()
        #     adj2_tensor = adj2
        #     if adj1_tensor.is_sparse:
        #         adj1_tensor = adj1_tensor.to_dense()
        #     if adj2_tensor.is_sparse:
        #         adj2_tensor = adj2_tensor.to_dense()
        #     wg = self.agcn_ag( torch.cat((adj1_tensor,adj2_tensor), 1) )
        #     wg = F.normalize(wg,p=2)
        #     wg1 = torch.reshape(wg[:,0], [n_x, 1])
        #     wg2 = torch.reshape(wg[:,1], [n_x, 1])
        #     wg1_broadcast = wg1.repeat(1,n_x)
        #     wg2_broadcast = wg2.repeat(1,n_x)
        #     adj = wg1_broadcast.mul(adj1_tensor) + wg2_broadcast.mul(adj2_tensor)
        ## [end] 边重定义

        masked_edge_loss = None

        if apply_edge_mask and masked_edges is not None:
            # 高效张量处理 masked_edges
            mask_idx = torch.tensor(list(masked_edges), dtype=torch.long).t().to(adj.device)  # shape: [2, num_edges]

            # 克隆一份邻接矩阵用于修改（避免原地改动）
            adj = adj.clone()

            # 使用 tensor 索引一次性掩码所有边（包括对称边）
            adj[mask_idx[0], mask_idx[1]] = 0
            adj[mask_idx[1], mask_idx[0]] = 0

        # 编码阶段
        feat, z_igae_adj = self.encoder(x, adj)

        # # 解码阶段（是否使用自表达）
        # if self.use_se:
        #     latent_recon = self.self_expr_layer(feat)
        #     recon, z_hat_adj = self.decoder(latent_recon, adj)
        # else:
        #     latent_recon = None
        #     recon, z_hat_adj = self.decoder(feat, adj)
        #
        # # 聚类头输出 soft assignment
        # q = self.cluster_layer(feat)
        #
        # # 结构重建损失：只在 masked 边上计算
        # if apply_edge_mask and masked_edges is not None:
        #     masked_edge_loss = F.mse_loss(
        #         z_hat_adj[mask_idx[0], mask_idx[1]],
        #         adj[mask_idx[0], mask_idx[1]]
        #     )

        # return feat, recon, latent_recon, z_hat_adj, q, masked_edge_loss

        return feat


class MultiViewModel(nn.Module):

    def __init__(self, in_channels, gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, gae_n_dec_1, gae_n_dec_2, gae_n_dec_3,
                 n_samples, n_clusters, use_se, topk, if_mask, device):
        super(MultiViewModel, self).__init__()
        self.aes = []
        for in_c in in_channels:
            self.aes.append(SingleViewModel(in_c,
                                            gae_n_enc_1, gae_n_enc_2, gae_n_enc_3,
                                            gae_n_dec_1, gae_n_dec_2, gae_n_dec_3,
                                            n_samples, n_clusters, use_se, topk, if_mask, device))
        self.aes = nn.ModuleList(self.aes)

        self.hidden_dims = gae_n_enc_3
        self.structure_decoder = EdgeDecoder(
            in_channels=self.hidden_dims,
            # hidden_channels=args.decoder_channels,
            hidden_channels=64,
            out_channels=1,
            # num_layers=args.structure_decoder_layer,
            num_layers=1,
            # dropout=args.structure_decoder_dropout,
            dropout=0.4,
            activation='relu'
        )

    def forward(self, x, adj, epoch, T, init_rate=0.5):
        output = []
        qs = []
        q_fulls = []
        for i in range(len(x)):
            feat, recon, latent_recon, z_hat_adj, prob, feat1, q_full = self.aes[i](x[i], adj[i], epoch=epoch, T=T,
                                                                                    init_rate=init_rate)
            qs.append(prob)
            q_fulls.append(q_full)
            output.append([feat, recon, latent_recon, z_hat_adj, feat1])

        return output, qs, q_fulls

    # def generate_masked_edges_union(self, x_list, adj_list, knn_k=50):
    #     all_cluster_edges = []
    #     for i in range(len(x_list)):
    #         edges = self.aes[i].get_cluster_edges(x_list[i], adj_list[i])
    #         all_cluster_edges.append(edges)
    #     Af = set.union(*all_cluster_edges)
    #
    #     # 用第0个视图生成KNN边集（你也可以改成多个视图KNN交集）
    #     knn_edges = self.aes[0].build_knn_graph(x_list[0], k=knn_k)
    #
    #     return Af.intersection(knn_edges)

    def generate_masked_edges_union(self, x_list, adj_list, knn_k=50):
        # [改进] 快速合并所有边集合
        all_cluster_edges = (
            self.aes[i].get_cluster_edges(x_list[i], adj_list[i])
            for i in range(len(x_list))
        )
        Af = set(chain.from_iterable(all_cluster_edges))

        # [可选缓存] 仅用第0个视图构建KNN边（也可加缓存）
        knn_edges = self.aes[0].build_knn_graph(x_list[0], k=knn_k)

        return Af.intersection(knn_edges)

    # def decode_edges(self, z, edge_index):
    #     z_i = z[edge_index[:, 0]]
    #     z_j = z[edge_index[:, 1]]
    #     return torch.sigmoid(self.structure_decoder(torch.cat([z_i, z_j], dim=1))).squeeze()

    def decode_edges(self, z, edge_index):
        return self.structure_decoder(z, z, edge_index, sigmoid=True).squeeze()

    def ce_loss(self, pos_score, neg_score):
        pos_loss = -torch.log(pos_score + 1e-8).mean()
        neg_loss = -torch.log(1 - neg_score + 1e-8).mean()
        return pos_loss + neg_loss

    def edge_loss(self, views, graphs, masked_features):
        loss = views[0].new_zeros(())
        for branch, inputs, graph, features in zip(self.aes, views, graphs, masked_features):
            with torch.no_grad():
                original, _ = branch.encoder(inputs, graph)
                pairs = ((graph != 0) | (graph.T != 0)).triu(1).nonzero().T
                homogeneous = torch.ones(pairs.size(1), device=graph.device, dtype=torch.bool)
                for restart in range(3):
                    km = PyTorchKMeans(n_clusters=branch.cluster_layer.cluster_centers.size(0),
                                       n_init=1, random_state=restart, verbose=False)
                    labels = km.fit_predict(original)
                    homogeneous &= labels[pairs[0]] == labels[pairs[1]]
            masked_graph = graph.clone()
            row, col = pairs[:, homogeneous]
            masked_graph[row, col] = 0
            masked_graph[col, row] = 0
            features, _ = branch.encoder(branch.edge_inputs, masked_graph)
            for selected, sign in ((homogeneous, -1), (~homogeneous, 1)):
                edges = pairs[:, selected]
                if edges.size(1):
                    partial = features.new_zeros(())
                    for chunk in edges.split(4096, dim=1):
                        logits = self.structure_decoder(features, features, chunk, sigmoid=False)
                        partial = partial + F.softplus(sign * logits).sum()
                    loss = loss + partial / edges.size(1)
        return loss
