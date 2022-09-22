import numpy as np
from numpy import ndarray
from matplotlib.pyplot import imsave
from skimage.color import rgb2gray
import time


class DIBArray:
    def __init__(self, img: ndarray):
        try:
            self.img = rgb2gray(img)
        except:
            self.img = img

    def save(self):
        imsave("../DIB-FrontEnd/src/assets/output/output.png",
               self.img, cmap='gray')

    def convolve(self, B, func):
        B = np.array(B)
        half = (int)((len(B) - 1) / 2)
        rows = len(self.img)
        pixels = len(self.img[0])
        ret = np.zeros([rows, pixels])

        start = time.perf_counter()
        for row in range(half, rows - half):
            for pixel in range(half, pixels - half):
                subArr = [arr[pixel - half: pixel + half + 1]
                          for arr in self.img[row - half: row + half + 1]]
                inters = np.logical_and(subArr, B)
                if func(np.mean(inters)):
                    ret[row][pixel] = 1
        end = time.perf_counter()
        print(end - start)
        return DIBArray(ret)

    def subArray(self, size, offset):
        x, y = offset
        return [arr[x - size: x + size + 1]
                for arr in self.img[y - size: y + size + 1]]

    def convolve2(self, B):
        B = np.array(B)
        half = (int)((len(B) - 1) / 2)
        rows = len(self.img)
        pixels = len(self.img[0])
        ret = np.zeros([rows, pixels])

        arr = self.img[half: rows - half]

        # [arr[] for row in arr]

        start = time.perf_counter()

        end = time.perf_counter()
        print(end - start)
        return DIBArray(ret)

    def __add__(self, B):
        return self.convolve(B, lambda mean: mean > 0)

    def __sub__(self, B):
        return self.convolve(B, lambda mean: mean == 1)


# print(im.img.shape)

# dim = 51
# shape = (dim, dim)
# half = (int)(dim / 2)
# quart = (int)(half / 2)
# img = np.zeros(shape, dtype=np.uint8)
# rr, cc = disk((half, half), quart, shape=shape)
# img[rr, cc] = 1
