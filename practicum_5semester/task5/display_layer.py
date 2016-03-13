import numpy as np
import PIL
import math

def display_layer(X, filename='layer.png'):
    pic_cnt = X.shape[0]

    pic_size = int(math.sqrt(X.shape[1]/3))
    cnt = math.ceil(math.sqrt(pic_cnt))
    n = cnt * (pic_size + 1) - 1
    res = np.zeros((n, n, 3))
    X1 = X.reshape(pic_cnt, pic_size, pic_size, 3)    
    minimum = np.min(X1, axis=(0, 1, 2))[np.newaxis, np.newaxis, np.newaxis, :]
    maximum = np.max(X1, axis=(0, 1, 2))[np.newaxis, np.newaxis, np.newaxis, :]
    X1 = (X1 - minimum) / (1e-5 + maximum - minimum)
    for i in range(pic_cnt):
        x = (i // cnt) * (pic_size + 1)
        y = (i % cnt) * (pic_size + 1)
        res[x : x+pic_size, y : y+pic_size, :] = X1[i]
    img = PIL.Image.fromarray(np.uint8(res * 255), 'RGB')
    img.resize((img.size[0] * 3, img.size[1] * 3)).save(filename)


#f=open('./data2.7/X1.pk', 'rb')
#images = picle.load(f)