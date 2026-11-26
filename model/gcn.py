import torch
from torch import nn
from torch_geometric.nn import GCNConv


# class IGAE_encoder(nn.Module):
#     '''
#     # 3层版
#     '''
#     def __init__(self, gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, n_input):
#         super(IGAE_encoder, self).__init__()
#         self.gnn_1 = GCNConv(n_input, gae_n_enc_1)
#         self.gnn_2 = GCNConv(gae_n_enc_1, gae_n_enc_2)
#         self.gnn_3 = GCNConv(gae_n_enc_2, gae_n_enc_3)
#         self.s = nn.Sigmoid()

#     def forward(self, x, adj):
#         adj = torch.tensor(adj.nonzero(), dtype=torch.long).t().contiguous()
#         z = self.gnn_1(x, adj)
#         z = self.gnn_2(z, adj)
#         z_igae = self.gnn_3(z, adj)
#         z_igae_adj = self.s(torch.mm(z_igae, z_igae.t()))
#         return z_igae, z_igae_adj


# class IGAE_decoder(nn.Module):

#     def __init__(self, gae_n_dec_1, gae_n_dec_2, gae_n_dec_3, n_input):
#         super(IGAE_decoder, self).__init__()
#         self.gnn_4 = GCNConv(gae_n_dec_1, gae_n_dec_2)
#         self.gnn_5 = GCNConv(gae_n_dec_2, gae_n_dec_3)
#         self.gnn_6 = GCNConv(gae_n_dec_3, n_input)
#         self.s = nn.Sigmoid()

#     def forward(self, z_igae, adj):
#         adj = torch.tensor(adj.nonzero(), dtype=torch.long).t().contiguous()
#         z = self.gnn_4(z_igae, adj)
#         z = self.gnn_5(z, adj)
#         z_hat = self.gnn_6(z, adj)
#         z_hat_adj = self.s(torch.mm(z_hat, z_hat.t()))
#         return z_hat, z_hat_adj



class IGAE_encoder(nn.Module):
    '''
    # 1层版
    '''

    def __init__(self, gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, n_input):
        super(IGAE_encoder, self).__init__()
        self.gnn_1 = GCNConv(n_input, gae_n_enc_3)
        self.s = nn.Sigmoid()

    def forward(self, x, adj):
        adj = torch.tensor(adj.nonzero(), dtype=torch.long).t().contiguous()
        z_igae = self.gnn_1(x, adj)
        z_igae_adj = self.s(torch.mm(z_igae, z_igae.t()))
        return z_igae, z_igae_adj


class IGAE_decoder(nn.Module):

    def __init__(self, gae_n_dec_1, gae_n_dec_2, gae_n_dec_3, n_input):
        super(IGAE_decoder, self).__init__()
        self.gnn_4 = GCNConv(gae_n_dec_1, n_input)
        self.s = nn.Sigmoid()

    def forward(self, z_igae, adj):
        adj = torch.tensor(adj.nonzero(), dtype=torch.long).t().contiguous()
        z_hat = self.gnn_4(z_igae, adj)
        z_hat_adj = self.s(torch.mm(z_hat, z_hat.t()))
        return z_hat, z_hat_adj


class IGAE(nn.Module):

   def __init__(self, gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, gae_n_dec_1, gae_n_dec_2, gae_n_dec_3, n_input):
       super(IGAE, self).__init__()
       self.encoder = IGAE_encoder(
           gae_n_enc_1=gae_n_enc_1,
           gae_n_enc_2=gae_n_enc_2,
           gae_n_enc_3=gae_n_enc_3,
           n_input=n_input)

       self.decoder = IGAE_decoder(
           gae_n_dec_1=gae_n_dec_1,
           gae_n_dec_2=gae_n_dec_2,
           gae_n_dec_3=gae_n_dec_3,
           n_input=n_input)

   def forward(self, x, adj):
       z_igae, z_igae_adj = self.encoder(x, adj)
       z_hat, z_hat_adj = self.decoder(z_igae, adj)
       adj_hat = z_igae_adj + z_hat_adj
       return z_igae, z_hat, adj_hat

   def __str__(self):
       return f"IGAE(encoder={self.encoder}, decoder={self.decoder})"

   def __repr__(self):
       return self.__str__()
