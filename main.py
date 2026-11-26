import os
import random

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim import Adam
from time import time
from copy import deepcopy

import warnings
warnings.filterwarnings('ignore')

from dataset import MMDataset, get_config
from utils.evaluation import cluster_accuracy, get_parameter_number, purity_score, draw_clu
from utils.postprocess import spixel_to_pixel_labels
from torch_clustering import PyTorchKMeans
from model.multiview_for_train_mm2 import MultiViewModel


from pathlib import Path
import sys
PROJECT_DIR = Path(__file__).resolve().parent
config = get_config(PROJECT_DIR / 'config.json')
os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu_id']
if config['dataset_name'] != 'Trento':
    color_tabel = np.array([[0, 0, 0],
                            [110, 240, 70], [0, 0, 243], [230, 50, 30], [70, 157, 71], [230, 50, 240],
                            [255, 255, 84], [83, 183, 223], [227, 203, 244], [130, 80, 180],
                            [10, 35, 78], [170, 50, 100], [113, 180, 50], [200, 100, 60],
                            [60, 128, 161], [210, 146, 60], [84, 185, 54], [122, 75, 50], ])
else:
    color_tabel = np.array([[0, 0, 0],
                            [56, 108, 53], [228, 55, 38], [230, 240, 85], [170, 228, 49],
                            [147, 75, 48], [0, 204, 0], ])


color_tabel = color_tabel / 255.0



def evaluate(y_pre_sp, association_mat, gt):
    if isinstance(y_pre_sp, torch.Tensor):
        y_pre_sp = y_pre_sp.cpu().numpy()
    if isinstance(association_mat, torch.Tensor):
        association_mat = association_mat.cpu().numpy()
    y_pre_pixel = spixel_to_pixel_labels(y_pre_sp, association_mat)
    class_map = y_pre_pixel.reshape(gt.shape)

    indx = np.where(gt != 0)
    y_target = gt[indx]
    y_predict = class_map[indx]
    y_predict, acc, kappa, nmi, ari, pur, ca = cluster_accuracy(y_target, y_predict, return_aligned=True)
    return (acc, kappa, nmi, ari, pur), ca, (indx, y_target.astype(int), y_predict)


def print_table(data, headers, col_width=12):
    num_cols = len(headers)
    col_spacing = 3
    
    row_format = "".join(["{:>"+str(col_width)+"}"]*(num_cols))
    row_separator = "".join(["{:->"+str(col_width)+"}"]*(num_cols))
    
    print(row_format.format(*headers))
    print(row_separator.format(*["" for _ in range(num_cols)]))
    
    for i, row in enumerate(data):
        row_values = [f"{val:.4f}" if isinstance(val, float) else val for val in row]
        print(row_format.format(*row_values))


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def compute_target_distribution(q_list):
    """
    For a list of q tensors from each view, compute list of P.
    """
    P_list = []
    for q in q_list:
        P = target_distribution(q)
        P_list.append(P)
    return P_list

def kl_divergence(p, q):
    """
    KL divergence between two distributions.
    """
    return (p * (p.log() - q.log())).sum(dim=1).mean()


def compute_version2_loss(q_full_list, q_masked_list, output, kl_divergence_fn):
    """
    Version 2 loss: KL(P || q_masked) + KL(P || H)
    where H is softmax(feat1)
    """
    P_list = compute_target_distribution([q.detach() for q in q_masked_list])
    kl_loss = 0
    for P, q_masked, out in zip(P_list, q_masked_list, output):
        feat1 = out[4]  # out = [feat, recon, latent_recon, z_hat_adj, feat1]
        log_h = F.log_softmax(feat1, dim=1)
        kl_loss += kl_divergence_fn(P, q_masked)
        kl_loss += F.kl_div(log_h, P, reduction='batchmean')

    return kl_loss


def train(config, pretrain=True):
    dataset_name = config['dataset_name']
    dataset = MMDataset(
        dataset_name, 
        config[dataset_name]['mm_data_path'], 
        config[dataset_name]['gt_path'],
        config[dataset_name]['n_pc'],
        config[dataset_name]['patch_size'],
        config[dataset_name]['rgb'],
        config[dataset_name]['n_neighbors'],
        config[dataset_name]['n_superpixels'],
    )
    print(dataset)
    if config['remove_bkg']:
        dataset.remove_background()
    n_clusters = dataset.n_classes
    n_samples = len(dataset)
    lr = config['lr']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCH = config['n_epoches']
    
    gae_n_enc_1, gae_n_enc_2, gae_n_enc_3 = config['gae_n_enc_1'], config['gae_n_enc_2'], config['gae_n_enc_3']
    gae_n_dec_1, gae_n_dec_2, gae_n_dec_3 = config['gae_n_dec_1'], config['gae_n_dec_2'], config['gae_n_dec_3']

    refine_k, rate = config[dataset_name]['refine_k'], config[dataset_name]['rate']     # 控制簇内对比学习比例
    alpha_recon, alpha_consist, alpha_contrast = config[dataset_name]['alpha_recon'], config[dataset_name]['alpha_consist'], config[dataset_name]['alpha_contrast']
    beta_edge_recon = config[dataset_name]['beta_edge_recon']
    beta_kl = config[dataset_name]['beta_kl']
    init_rate = config[dataset_name]['init_rate']
    use_se = True       # 使用自表达层
    if_draw = config['if_draw']
    if_mask = config['if_mask']

    save_path = config['save_path'] + '/' + dataset_name + '.pkl'
    n_views = len(dataset.x)

    x = [torch.from_numpy(it.reshape(n_samples, -1)).float().to(device) for it in dataset.x]
    adj = [torch.from_numpy(it).float().to(device) for it in dataset.adj]

    y = dataset.y
    model = MultiViewModel(
        in_channels=[it.shape[1] for it in x],
        gae_n_enc_1=gae_n_enc_1,
        gae_n_enc_2=gae_n_enc_2,
        gae_n_enc_3=gae_n_enc_3,
        gae_n_dec_1=gae_n_dec_1,
        gae_n_dec_2=gae_n_dec_2,
        gae_n_dec_3=gae_n_dec_3,
        n_samples=n_samples,
        n_clusters=n_clusters,
        use_se=use_se,
        topk=refine_k,
        if_mask = if_mask,
        device=device).to(device)
    
    get_parameter_number(model)

    optimizer = Adam(model.parameters(), lr=lr)
    kmeans = PyTorchKMeans(n_clusters=n_clusters, init='k-means++', distributed=True, verbose=False)


    z = torch.cat(x, dim=1)         # 不同模态cat起来的
    y_pred = kmeans.fit_predict(z).cpu().numpy()
    if config['remove_bkg']:
        y_pred = dataset.recover_background(y_pred)
    mts, ca, eta = evaluate(y_pred, dataset.association_mat, dataset.gt)
    print('[initial accuracy]')
    print_table([mts], ['ACC', 'Kappa', 'NMI', 'ARI', 'Purity'])
    print()

    t0 = time()
    if pretrain:
        print('start training.')
        best_acc = [0] * 5
        for epoch in range(1, EPOCH + 1):
            model.train()
            output, qs, q_fulls = model(x, adj, epoch=epoch, T=EPOCH, init_rate=init_rate)    # output: feat, recon, latent_recon, adj_hat
            edge_recon_loss = model.edge_loss(x, adj, [view[0] for view in output])


            recon_loss = x[0].new_zeros(())
            for v in range(n_views):
                masked = model.aes[v].masked_nodes
                if masked.any():
                    cosine = F.cosine_similarity(output[v][1][masked], x[v][masked])
                    recon_loss += (1 - cosine).clamp_min(0).pow(3).mean()

            loss = 0
            loss += alpha_recon * recon_loss
            loss += beta_edge_recon * edge_recon_loss

            kl_loss_v2 = compute_version2_loss(q_fulls, qs, output, kl_divergence)
            loss += beta_kl * kl_loss_v2
            

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(
                f'Epoch {epoch}: loss = {loss.item()}, recon_loss = {recon_loss * alpha_recon}, edge_recon_loss = {beta_edge_recon * edge_recon_loss}, kl_loss = {beta_kl * kl_loss_v2} ')
            
            model.eval()
            with torch.no_grad():
                output, qs, q_fulls = model(x, adj, epoch=epoch, T=EPOCH, init_rate=init_rate)
            z = torch.cat([output[v][0] for v in range(n_views)], dim=1) # 不同模态cat起来的
            y_pred = kmeans.fit_predict(z).cpu().numpy()
            if config['remove_bkg']:
                y_pred = dataset.recover_background(y_pred)
            metrics, ca, eta = evaluate(y_pred, dataset.association_mat, dataset.gt)
            if metrics[0] > best_acc[0]:
                best_acc = metrics
                weights = deepcopy(model.state_dict())
                indx, true, pred = eta
                if if_draw == True:
                    draw_clu(dataset.gt, true, pred, dataset_name,
                             './picture' + '/' + dataset_name + '{}.png'.format(best_acc[0]))

            print_table([['Best', *best_acc], ['Current', *metrics]],
                        ['Result', 'ACC', 'Kappa', 'NMI', 'ARI', 'Purity'])
        torch.save(weights, save_path)
        print(f'best weight saved in {save_path}')
        print('finish.')

    
    return best_acc, time() - t0, ca


def rand_seed(seed):
    if seed == -1:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    os.chdir(PROJECT_DIR)
    name = sys.argv[1] if len(sys.argv) > 1 else 'CG-MAE'
    if config['if_draw']:
        Path('picture').mkdir(exist_ok=True)
    rows = []
    for seed in config['seeds']: # 0-4
        rand_seed(seed)
        run = dict(config)
        run['save_path'] = str(Path(config['save_path']) / name / f'seed_{seed}')
        Path(run['save_path']).mkdir(exist_ok=True, parents=True)
        Path('result').mkdir(exist_ok=True)
        print(f'Seed {seed}')
        metrics, elapsed, _ = train(run)
        rows.append(metrics)
    table = [[str(seed), *row] for seed, row in zip(config['seeds'], rows)]
    if len(rows) > 1:
        table.extend([['Mean', *np.mean(rows, axis=0)],
                      ['Sample SD', *np.std(rows, axis=0, ddof=1)]])
    print_table(table, ['Seed', 'ACC', 'Kappa', 'NMI', 'ARI', 'Purity'])
