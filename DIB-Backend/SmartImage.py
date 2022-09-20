import numpy as np
from math import sqrt
from numpy.fft import fft2, ifft2
from skimage.filters import threshold_sauvola, threshold_otsu, threshold_mean
from skimage.color import rgb2gray, gray2rgb
from skimage.restoration import denoise_bilateral
from skimage import feature
from scipy.ndimage import maximum_filter, minimum_filter, gaussian_filter, sobel
from scipy.signal import convolve2d, gaussian
from PIL import Image
from matplotlib.pyplot import imsave


class SmartImage:
    def __init__(self, img: np.ndarray):
        self.img = img

    def gaussian_kernel(kernel_size=3):
        h = gaussian(kernel_size, kernel_size / 3).reshape(kernel_size, 1)
        h = np.dot(h, h.transpose())
        h /= np.sum(h)
        return h

    def wiener(self, kernel=gaussian_kernel(), K=10):
        # normalise kernel
        kernel /= np.sum(kernel)
        # fft of spatial degradation
        kernel = fft2(kernel, s=self.img.shape)
        # core formula for wiener
        kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
        # fft of copy of image
        copy = fft2(np.copy(self.img))
        # calculate g(x) = f(x) * h(x)
        copy *= kernel
        # return to spatial domain
        self.img = np.abs(ifft2(copy))
        return self

    def contrast(self):
        epsilon = 0.00001 * np.ones(self.img.shape)
        Max = maximum_filter(self.img, size=3)
        Min = minimum_filter(self.img, size=3)
        self.img = (Max - Min) / (Max + Min + epsilon)
        return self

    def contrast_gradient(self, power=1):
        alpha = sqrt(np.var(self.img)) ** power
        epsilon = 0.00001 * np.ones(self.img.shape)
        Max = maximum_filter(self.img, size=3)
        Min = minimum_filter(self.img, size=3)
        dif = (Max - Min)
        self.img = alpha * (dif / (Max + Min + epsilon)) + (1 - alpha) * dif
        return self

    def rgb2gray(self):
        self.img = rgb2gray(self.img)
        return self

    def gray2rgb(self):
        self.img = gray2rgb(self.img)
        return self

    # def extract(self):
    #     mean = np.mean(self.img)
    #     stdev = sqrt(np.var(self.img)) / 2
    #     thresh = np.zeros(self.img.shape)
    #     thresh[self.img <= mean + stdev] = 1
    #     self.img = thresh
    #     return self

    def denoise(self):
        try:
            self.img = denoise_bilateral(self.img, channel_axis=-1)
        except:
            self.img = denoise_bilateral(self.img)
        return self

    def sauvola(self):
        self.img = self.img > threshold_sauvola(self.img)
        return self

    def otsu(self):
        self.img = self.img > threshold_otsu(self.img)
        return self

    def meanThreshold(self):
        self.img = self.img > threshold_mean(self.img)
        return self

    def canny(self):
        self.img = feature.canny(self.img)
        return self

    def sobel(self, axis=0):
        self.img = sobel(self.img, axis)
        return self

    def sobelMag(self):
        xdir = sobel(self.img, axis=0, mode='constant')
        ydir = sobel(self.img, axis=1, mode='constant')
        # self.img = np.hypot(xdir, ydir)
        return self

    def invert(self):
        self.img = np.ones(self.img.shape) - self.img
        return self

    def blur(self, sigma=1):
        self.img = gaussian_filter(self.img, sigma)
        return self

    def sharpen(self, alpha=0.8, sigma=1):
        blurred = gaussian_filter(self.img, sigma)
        self.img = self.img + alpha * (self.img - blurred)
        return self

    def convolve_filter(self, kernel=gaussian_kernel()):
        self.img = convolve2d(self.img, kernel, 'same', boundary='fill',
                              fillvalue=0)
        return self

    def save(self):
        imsave("../DIB-FrontEnd/src/assets/output/output.png",
               self.img, cmap='gray')


def edgeArray(img: SmartImage):
    widths = []
    for row in img.img:
        length = 0
        for pixel in row:
            if pixel:
                length += 1
            else:
                if length:
                    widths.append(length)
                length = 0
    return widths


def edges(original, edge):
    widths = []
    newImg = np.zeros(original.shape)
    for i, row in enumerate(edge):
        # for i in range(100):
        # blackPixels = 0
        count = 0
        for j in range(len(edge[0]) - 2):
            if (not edge[i][j] and edge[i][j+1]):
                newImg[i][j+1] = 1
                # blackPixels = 0
                if (count != 0):
                    widths.append(count)
                count = 0
                # continue

            # if (edge[i][j] != original[i][j] and blackPixels < 2):
            if (edge[i][j] < original[i][j]):
                newImg[i][j] = 0.3
                count += 1
                # continue

            # blackPixels += 1
        if (count != 0):
            widths.append(count)
    return newImg, widths
