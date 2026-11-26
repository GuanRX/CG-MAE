from warnings import simplefilter

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from sklearn import metrics
from munkres import Munkres

def evaluate(label, pred, is_HSI_score=False):
    # if not is_labeled_pixel:
    #     indx_nonzero = np.where(label != 0)
    #     pred = pred[indx_nonzero]
    #     label = label[indx_nonzero]
    nmi = metrics.normalized_mutual_info_score(label, pred)
    ari = metrics.adjusted_rand_score(label, pred)
    f = metrics.fowlkes_mallows_score(label, pred)
    pred_adjusted = get_y_preds(label, pred, len(set(label)))
    acc = metrics.accuracy_score(pred_adjusted, label)
    if is_HSI_score:
        pur = purity_score(label, pred_adjusted)
        ca = class_acc(label, pred_adjusted)
        kappa = metrics.cohen_kappa_score(label, pred_adjusted)
        return acc, kappa, nmi, ari, pur, ca
    return nmi, ari, f, acc


def calculate_cost_matrix(C, n_clusters):
    cost_matrix = np.zeros((n_clusters, n_clusters))
    # cost_matrix[i,j] will be the cost of assigning cluster i to label j
    for j in range(n_clusters):
        s = np.sum(C[:, j])  # number of examples in cluster i
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s - t
    return cost_matrix


def get_cluster_labels_from_indices(indices):
    n_clusters = len(indices)
    cluster_labels = np.zeros(n_clusters)
    for i in range(n_clusters):
        cluster_labels[i] = indices[i][1]
    return cluster_labels


def get_y_preds(y_true, cluster_assignments, n_clusters):
    """
    Computes the predicted labels, where label assignments now
    correspond to the actual labels in y_true (as estimated by Munkres)
    cluster_assignments:    array of labels, outputted by kmeans
    y_true:                 true labels
    n_clusters:             number of clusters in the dataset
    returns:    a tuple containing the accuracy and confusion matrix,
                in that order
    """
    confusion_matrix = metrics.confusion_matrix(y_true, cluster_assignments, labels=None)
    # compute accuracy based on optimal 1:1 assignment of clusters to labels
    cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters)
    indices = Munkres().compute(cost_matrix)
    kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)

    if np.min(cluster_assignments) != 0:
        cluster_assignments = cluster_assignments - np.min(cluster_assignments)
    y_pred = kmeans_to_true_cluster_labels[cluster_assignments]
    return y_pred


def class_acc(y_true, y_pre):
    """
    calculate each class's acc
    :param y_true:
    :param y_pre:
    :return:
    """
    ca = []
    for c in np.unique(y_true):
        y_c = y_true[np.nonzero(y_true == c)]  # find indices of each classes
        y_c_p = y_pre[np.nonzero(y_true == c)]
        acurracy = metrics.accuracy_score(y_c, y_c_p)
        ca.append(acurracy)
    ca = np.array(ca)
    return ca


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

def best_match(y_true, y_pre):
    Label1 = np.unique(y_true)
    nClass1 = len(Label1)
    Label2 = np.unique(y_pre)
    nClass2 = len(Label2)
    if nClass1 < nClass2:       # 默认y_true的标签一定齐全一点
        Label1 = Label2
        nClass1 = nClass2
    nClass = np.maximum(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        ind_cla1 = y_true == Label2[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = y_pre == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:, 1]
    y_best = np.zeros(y_pre.shape)
    for i in range(nClass2):
        y_best[y_pre == Label2[i]] = Label1[c[i]]
    return y_best

# # Alternative version
def cluster_accuracy(y_true, y_pre, return_aligned=False):
    y_true = y_true.astype('float32')
    y_pre = y_pre.astype('float32')
    Label1 = np.unique(y_true)
    nClass1 = len(Label1)
    Label2 = np.unique(y_pre)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        ind_cla1 = y_true == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = y_pre == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:, 1]
    y_best = np.zeros(y_pre.shape)
    for i in range(nClass2):
        y_best[y_pre == Label2[i]] = Label1[c[i]]

    # # calculate accuracy
    err_x = np.sum(y_true[:] != y_best[:])
    missrate = err_x.astype(float) / (y_true.shape[0])
    acc = 1. - missrate
    nmi = metrics.normalized_mutual_info_score(y_true, y_pre)
    kappa = metrics.cohen_kappa_score(y_true, y_best)
    ca = class_acc(y_true, y_best)
    ari = metrics.adjusted_rand_score(y_true, y_best)
    fscore = metrics.f1_score(y_true, y_best, average='micro')
    pur = purity_score(y_true, y_best)
    if return_aligned:
        return y_best, acc, kappa, nmi, ari, pur, ca
    return acc, kappa, nmi, ari, pur, ca


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('=====Model Parameters====')
    print('Total:{:.3f}M,\nTrainable: {:.3f}M'.format(total_num/1e6, trainable_num/1e6))

def draw_clu(gt, y, ids, dataname, save_path=None):
    # gt = gt.T
    h, w = gt.shape
    # N = np.max(y) + 1  # Get the maximum cluster label value
    N = 20
    ids = np.array(ids, dtype=int)
    # ids = ids + 1
    # y += 1
    g = np.unique(gt)
    Y = np.unique(y)
    i = np.unique(ids)
    print('g=', g)
    print('y=', Y)
    print('i=', i)

    # y += 1
    # if dataname == 'Muu':
    #     gt += 1
    # g = np.unique(gt)
    # Y = np.unique(y)
    # i = np.unique(ids)
    # print('g=', g)
    # print('y=', Y)
    # print('i=', i)

    # Use a built-in color map in Matplotlib
    # color_map = plt.cm.jet(np.linspace(0, 1, N))[:3]
    color_map = [color[:3] for color in plt.cm.jet(np.linspace(0, 1, N))]
    color_map[0] = [0, 0, 0]  # Set label 0's color to black
    # Define other specific colors
    if dataname == 'Trento' or dataname == 'Augsburg':
        custom_colors = {
            2: [228, 55, 38],
            1: [56, 108, 53],
            3: [230, 240, 85],
            4: [170, 228, 49],
            5: [147, 75, 48],
            6: [0, 204, 0],
            7: [128, 255, 255]
        }
    else:
        custom_colors = {
            1: [110, 240, 70],
            2: [0, 0, 243],
            3: [230, 50, 30],
            5: [230, 50, 240],
            4: [70, 157, 71],
            6: [255, 255, 84],
            7: [83, 183, 223],
            8: [227, 203, 244],
            9: [130, 80, 180],
            11: [170, 50, 100],
            10: [10, 35, 78],
            12: [113, 180, 50],
            13: [200, 100, 60],
            14: [60, 128, 161],
            15: [210, 146, 60],
            17: [122, 75, 50],
            16: [84, 185, 54]
        }
    for label, color in custom_colors.items():
        color_map[label] = np.array(color) / 255.0

    # Initialize the RGB image
    rgb_image = np.zeros((h, w, 3))

    # Assign colors to each pixel
    for m in range(h):
        for n in range(w):
            rgb_image[m, n] = color_map[int(gt[m, n])]

    # Display the clustering result image
    plt.figure()
    plt.imshow(rgb_image)
    plt.axis('off')

    # Create a confusion matrix
    C = confusion_matrix(y, ids)

    # Initialize the mapping dictionary
    label_mapping = {}
    #
    # For each cluster label, find the corresponding most common true label
    for n in range(C.shape[1]):
        max_idx = np.argmax(C[:, n])
        label_mapping[n + 1] = max_idx + 1  # Adjust to match labels starting from 0

    print(f"label_mapping: {label_mapping}")  # Add this line for debugging

    # Update ids with the mapped labels
    for n in range(len(ids)):
        ids[n] = label_mapping[ids[n]]

    # rgb_image = np.zeros((h, w, 4))
    #
    # # Assign colors for each pixel based on the new labels
    # cnt = 0
    # # pre = ids.reshape(h, w)
    # for m in range(h):
    #     for n in range(w):
    #         if gt[m, n] != 0:
    #             label = ids[cnt]
    #             # label = pre[m, n]
    #             rgb_image[m, n] = color_map[label]
    #             cnt += 1
    #         else:
    #             rgb_image[m, n] = color_map[0]
    #
    # # Display the final clustering result image
    # plt.figure()
    # plt.imshow(rgb_image)
    # plt.axis('off')
    #
    # plt.show()
    fig_true = np.zeros((h, w, 3))
    fig_pred = np.zeros((h, w, 3))
    # color_map = color_tabel / 256.0
    colors_true = [color_map[y[i]] for i in range(len(y))]
    print("true:", y)
    colors_pred = [color_map[ids[i]] for i in range(len(ids))]

    points = np.argwhere(gt != 0)
    for i in range(points.shape[0]):
        p = points[i]
        fig_true[p[0], p[1], :] = colors_true[i]
        fig_pred[p[0], p[1], :] = colors_pred[i]

    # plt.figure(figsize=(10, 7))
    # ax1 = plt.subplot(1, 2, 1)
    # ax1.imshow(fig_true)
    # ax1.set_title('true')
    # ax2 = plt.subplot(1, 2, 2)
    # ax2.imshow(fig_pred)
    # ax2.set_title('predict')
    # 创建一个新的图形（使用默认大小）

    # fig_pred = np.fliplr(fig_pred)

    plt.figure()
    plt.imshow(fig_pred)  # 显示预测聚类的图像
    plt.axis('off')  # 关闭坐标轴

    if save_path is not None:
        plt.savefig(save_path, dpi=1500, bbox_inches='tight', pad_inches=0.0)
        print('### Save figure in {}. ###'.format(save_path))

    color = np.unique(colors_true)
    # print(color)
    # plt.show()
    plt.close()

    return fig_pred