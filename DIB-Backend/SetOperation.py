import numpy as np
from numpy import ndarray
from matplotlib.pyplot import imsave
from SmartImage import SmartImage
import time


class SetOperation:
    def __init__(self, img: np.ndarray):
        self.img = SmartImage(img).rgb2gray().img
        # self.img = img

    def save(self):
        imsave("../DIB-FrontEnd/src/assets/output/output.png",
               self.img, cmap='gray')

    def dilate(self, size=3):
        half = (int)((size - 1) / 2)
        rows = len(self.img)
        pixels = len(self.img[0])
        ret = np.zeros([rows, pixels])
        start = time.time()
        for row in range(half, rows - half):
            for pixel in range(half, pixels - half):
                b = False
                for i in range(-half, half):
                    for j in range(-half, half):
                        if self.img[row + i][pixel + j] > 0:
                            b = True
                if b:
                    ret[row][pixel] = 1
        self.img = ret
        end = time.time()
        print(f"execution time: {end - start}")
        return self

    def dilate1(self, size=3):
        half = (int)((size - 1) / 2)
        rows = len(self.img)
        pixels = len(self.img[0])
        ret = np.zeros([rows, pixels])
        B = np.array([[1 for i in range(size)] for j in range(size)])
        start = time.time()
        for row in range(half, rows - half):
            for pixel in range(half, pixels - half):
                subArr = [arr[pixel - half: pixel + half + 1]
                          for arr in self.img[row - half: row + half + 1]]
                inters = np.logical_and(subArr, B)
                if np.linalg.norm(inters) > 0:
                    ret[row][pixel] = 1
        self.img = ret
        end = time.time()
        print(f"execution time: {end - start}")
        return self

    def __add__(self, B):
        B = np.array(B)
        half = (int)((len(B) - 1) / 2)
        rows = len(self.img)
        pixels = len(self.img[0])
        ret = np.zeros([rows, pixels])

        for row in range(half, rows - half):
            for pixel in range(half, pixels - half):
                subArr = [arr[pixel - half: pixel + half + 1]
                          for arr in self.img[row - half: row + half + 1]]
                inters = np.logical_and(subArr, B)
                if np.linalg.norm(inters) > 0:
                    ret[row][pixel] = 1
        self.img = ret
        return self


class DIBArray(ndarray):
    def __init__(self, img: ndarray):
        self.img = img

    def save(self):
        imsave("../DIB-FrontEnd/src/assets/output/output.png",
               self.img, cmap='gray')

    def __add__(self, B):
        B = np.array(B)
        half = (int)((len(B) - 1) / 2)
        rows = len(self.img)
        pixels = len(self.img[0])
        ret = np.zeros([rows, pixels])

        for row in range(half, rows - half):
            for pixel in range(half, pixels - half):
                subArr = [arr[pixel - half: pixel + half + 1]
                          for arr in self.img[row - half: row + half + 1]]
                inters = np.logical_and(subArr, B)
                if np.linalg.norm(inters) > 0:
                    ret[row][pixel] = 1
        self.img = ret
        return self
