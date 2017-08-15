import heapq
import numpy as np
import math
LATENT_CONCEPTS_NUM = 100
PERCENTAGE_PT = 0.8
STOP_ITERATOR_THRESHOLD = 20


def normalize(vec):
    s = sum(vec)
    for i in range(len(vec)):
        vec[i] = vec[i] * 1.0 / s


# k is the number of concepts
def calu_two_matrix(image_hist, k=LATENT_CONCEPTS_NUM):
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
        # normalize(p_w_z_new)
        # normalize(p_z_d_new)
        # normalize(p_z_d_w_new)
        p_z_d_w = p_z_d_w_new
        p_w_z = p_w_z_new
        p_z_d = p_z_d_new
        print 'diff is %g' % diff
        if diff < STOP_ITERATOR_THRESHOLD:
            break
    return p_w_z


# select visual word
def select_visual_word(p_w_z, pt=PERCENTAGE_PT):
    indexs = []
    shape = list(np.shape(p_w_z))
    # for every concept, find almost top 1-pt visual word
    for zi in range(shape[1]):
        visual_word_value = p_w_z[:, zi]
        target_num = int(shape[0] * (1.0 * (1-pt)))
        index = heapq.nlargest(target_num, range(shape[0]), visual_word_value.__getitem__)
        indexs.extend(index)
    return list(set(indexs))


# construct the selected hist image
def build_selected_hist_images(hist_images, p_w_z, pt=PERCENTAGE_PT):
    indexs = select_visual_word(p_w_z, pt)
    selected = hist_images[:, indexs]
    return selected