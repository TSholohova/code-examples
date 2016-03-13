import numpy as np


# v1_vector(img : 3d np.array of double, ch : 1d np.array of double)
#           img.shape[2] = ch.shape[0]
# returns 2d np.array of double
def v1_vector(img, ch):
    return np.sum(img * ch[np.newaxis, np.newaxis, :], axis=2)


# v2_non_vector(img : 3d np.array of double, ch : 1d np.array of double)
#           img.shape[2] = ch.shape[0]
# returns 2d np.array of double
def v2_non_vector(img, ch):
    h = img.shape[0]
    w = img.shape[1]
    res = np.zeros((h, w))
    for i in range(0, h):
        for j in range(0, w):
            for c in range(0, ch.shape[0]):
                res[i, j] += img[i, j, c] * ch[c]
    return res


# v3_part_vector(img : 3d np.array of double, ch : 1d np.array of double)
#           img.shape[2] = ch.shape[0]
# returns 2d np.array of double
def v3_part_vector(img, ch):
    h = img.shape[0]
    w = img.shape[1]
    res = np.zeros((h, w))
    for c in range(0, ch.shape[0]):
        res += img[:, :, c] * ch[c]
    return res


# gen(0 <= size <= 4: int) returns tuple(img : 3d np.array of double,
#                                       ch : 1d np.array of double)
def gen(size):
    shape_h = np.array([10, 20, 40, 80, 160])
    shape_w = np.array([15, 30, 60, 120, 240])
    shape_ch = np.array([3, 5, 10, 20, 50])
    img = np.random.randint(0, 5,
                            (shape_h[size], shape_w[size], shape_ch[size]))
    ch = np.random.uniform(-1.0, 1.0, shape_ch[size])
    return (img, ch)
