from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import load_input
import Tools
import math
import os
import SVM

KMEANS_CENTER_NUM = 300
MI_SELECT_NUM = 100


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

def normalize(vec):
    s = sum(vec)
    for i in range(len(vec)):
        vec[i] = vec[i] * 1.0 / s


# k is the number of concepts
def calu_two_matrix(image_hist, k=100):
    shape = list(np.shape(image_hist))
    p_w_z = np.zeros(
        [shape[1], k],
        dtype=np.float32
    )
    p_z_d = np.zeros(
        [k, shape[0]],
        dtype=np.float32
    )
    p_z_d_w = np.zeros(
        [k, shape[0], shape[1]],
        dtype=np.float32
    )
    p_z_d = np.random.random([k, shape[0]])
    for zi in range(k):
        normalize(p_z_d[zi])
    p_w_z = np.random.random([shape[1], k])
    for wi in range(shape[1]):
        normalize(p_w_z[wi])
    while True:
        # update p(z|d,w)
        p_z_d_w_new = np.zeros(
            [k, shape[0], shape[1]],
            dtype=np.float32
        )
        for zi in range(k):
            for di in range(shape[0]):
                for wi in range(shape[1]):
                    sum = 0.0
                    for zii in range(k):
                        sum += (p_w_z[wi, zii] + p_z_d[zii, di])
                    temp = p_w_z[wi, zi] * p_z_d[zi, di] * 1.0
                    temp /= (sum * 1.0)
                    p_z_d_w_new[zi, di, wi] = temp
        p_w_z_new = np.zeros(
            [shape[1], k],
            dtype=np.float32
        )
        p_z_d_new = np.zeros(
            [k, shape[0]],
            dtype=np.float32
        )
        # update p(w|z)
        sum_p_w_z_byk = {}
        for zi in range(k):
            sum_p_w_z = 0.0
            for wii in range(shape[1]):
                for dii in range(shape[0]):
                    sum_p_w_z += (image_hist[dii, wii] * p_z_d_w_new[zi, dii, wii])
            sum_p_w_z_byk[zi] = sum_p_w_z
        for wi in range(shape[1]):
            for zi in range(k):
                sum_up = 0.0
                for dii in range(shape[0]):
                    sum_up += (image_hist[dii, wi] * p_z_d_w_new[zi, dii, wi])
                sum_down = sum_p_w_z_byk[zi]
                temp = (sum_up * 1.0) / (sum_down * 1.0)
                p_w_z_new[wi, zi] = temp

        # update p(z,d)
        nd = []
        for dii in range(shape[0]):
            temp = 0.0
            for wii in range(shape[1]):
                temp += image_hist[dii, wii]
            nd.append(temp)
        for zi in range(k):
            for di in range(shape[0]):
                sum_up = 0.0
                for wii in range(shape[1]):
                    sum_up += (image_hist[di, wii] * p_z_d_w_new[zi, di, wii])
                temp = (sum_up * 1.0) / (nd[di])
                p_z_d_new[zi, di] = temp
        diff = math.fabs(
            np.sum(p_z_d_w_new - p_z_d_w)
        )
        p_z_d_w = p_z_d_w_new
        p_w_z = p_w_z_new
        p_z_d = p_z_d_new
        print 'diff is %g' % diff


if __name__ == '__main__':
    image_hist_npy_path = './data/image_hist.npy'
    if os.path.exists(image_hist_npy_path):
        images_hist = np.load(image_hist_npy_path)
        print 'load npy file successful'
    else:
        labels, patches, count = load_input.load_input()
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
    calu_two_matrix(images_hist)
    # SVM.do_svm(np.array(images_hist), np.array(labels))
