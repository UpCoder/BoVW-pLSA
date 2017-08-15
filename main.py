from sklearn.decomposition import PCA
import heapq
from sklearn.cluster import KMeans
import numpy as np
import load_input
import scipy.io as scio
import Tools
import math
import os
import pLSA
import SVM

KMEANS_CENTER_NUM = 300
LATENT_CONCEPTS_NUM = 100
PERCENTAGE_PT = 0.8


# do pca operation for X(patches)
# column_num is the param that determine the number of dimension after doing pca operation
def do_pca(x, column_num=10):
    pca = PCA(n_components=column_num)
    x = pca.fit_transform(x)
    return x


# calu hist
# count is the number of patches each image
# categories is pre patch's category
def calu_hist(count, categories, k_num=KMEANS_CENTER_NUM):
    hist = []
    start_index = 0
    for num in count:
        pre_hist = [0] * k_num
        for _ in range(num):
            pre_hist[categories[start_index]] += 1
            start_index += 1
        hist.append(pre_hist)
    return np.array(hist)


# will quantize the ith feature to L levels
def do_quantize(hist, levels=5):
    columns_num = np.shape(hist)[1]
    hist_out = np.zeros(np.shape(hist))
    for i in range(columns_num):
        one_column = hist[:, i]
        thresh = Tools.Tools.multithresh(one_column, levels)
        hist_out[:, i] = Tools.Tools.imquantize(one_column, thresh)
    return hist_out


def do_k_means(x, k_num=KMEANS_CENTER_NUM):
    kmeans_res = KMeans(n_clusters=k_num, random_state=0, n_jobs=-1).fit(x)
    return kmeans_res


def calu_all_labels(labels, count):
    res = []
    for index, item in enumerate(labels):
        res.extend([item] * count[index])
    return res


if __name__ == '__main__':
    image_hist_npy_path = './data/image_hist.npy'
    labels, patches, count = load_input.load_input()
    if os.path.exists(image_hist_npy_path):
        images_hist = np.load(image_hist_npy_path)
        print 'load npy file successful'
    else:
        all_labels = calu_all_labels(labels, count)
        patches = do_pca(patches)
        print 'finish pca operation, the patches shape is ', np.shape(patches)
        print 'all labels shape is ', np.shape(all_labels)
        kmeans_res = do_k_means(patches)
        images_hist = calu_hist(count, kmeans_res.labels_)
        np.save(
            image_hist_npy_path,
            images_hist
        )
        print 'save npy file successful'
    print 'image hist shape is ', np.shape(images_hist)
    pts = [0.8, 0.99]
    latent_concepts_nums = [(i + 1) * 10 for i in range(10)]
    # SVM.do_svm(np.array(images_hist), np.array(labels), n_flod=5, debug=False, is_linear=False)
    for pt in pts:
        for latent_concepts_num in latent_concepts_nums:
            print '-'*15, 'pt=', pt, ' latent_concepts_num=', latent_concepts_num, '-'*15
            save_path = './data/selected_image_hist_' + str(pt) + '_' + str(latent_concepts_num) + '.mat'
            if os.path.exists(save_path):
                selected_hist_images = scio.loadmat(save_path)['histogram']
            else:
                p_w_z = pLSA.calu_two_matrix(images_hist, k=latent_concepts_num)
                selected_hist_images = pLSA.build_selected_hist_images(images_hist, p_w_z, pt=pt)
            print 'selcted image hist shape is ', np.shape(selected_hist_images), ' remove rate is ',\
                1 - ((1.0 * np.shape(selected_hist_images)[1]) / (1.0 * np.shape(images_hist)[1]))
            scio.savemat(
                save_path,
                {
                    'histogram': selected_hist_images
                }
            )
            SVM.do_svm(np.array(selected_hist_images), np.array(labels), n_flod=5, debug=False, is_linear=False)
