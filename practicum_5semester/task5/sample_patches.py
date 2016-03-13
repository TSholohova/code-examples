import numpy as np


def normalize_data(images):
    std = np.std(images, axis=0)
    mean = np.mean(images, axis=0)
    images = np.minimum(images, (mean + 3 * std)[np.newaxis, :])
    images = np.maximum(images, (mean - 3 * std)[np.newaxis, :])
    minimum = np.min(images, axis=0)[np.newaxis, :]
    maximum = np.max(images, axis=0)[np.newaxis, :]
    return (0.1 + 0.8 * (images - minimum) / (maximum - minimum + 1e-5))


def sample_patches_raw(images, num_patches=10000, patch_size=8):
    pic_cnt = images.shape[0]
    pic_size = int(np.sqrt(images.shape[1]/3))
    nums_pic = np.random.randint(0, pic_cnt, num_patches)
    xs_pic = np.random.randint(0, pic_size-patch_size+1, num_patches)
    xs_pic = xs_pic[:, np.newaxis, np.newaxis] + \
             np.repeat(np.arange(patch_size), patch_size).reshape(patch_size, patch_size)
    ys_pic = np.random.randint(0, pic_size-patch_size+1, num_patches)
    ys_pic = ys_pic[:, np.newaxis, np.newaxis] + \
             np.tile(np.arange(patch_size), patch_size).reshape(patch_size, patch_size)
    patches = images[nums_pic].reshape(num_patches, pic_size, pic_size, 3)
    patches = patches[np.arange(num_patches)[:, np.newaxis, np.newaxis],
                      xs_pic, ys_pic, :]
    return patches.reshape(num_patches, 3 * patch_size * patch_size)


def sample_patches(images, num_patches=10000, patch_size=8):
    res = normalize_data(sample_patches_raw(images, num_patches, patch_size))
    return res