from re import X
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
from skimage.draw import disk
import time


# defB = [
#     [1, 1, 1],
#     [1, 1, 1],
#     [1, 1, 1]
# ]

# dim = 17
# B = [[1] * dim for j in range(dim)]

# dim = 17
# shape = (dim, dim)
# half = (int)(dim / 2)
# quart = (int)(half / 2)
# img = np.zeros(shape, dtype=np.uint8)
# rr, cc = disk((half, half), quart, shape=shape)
# img[rr, cc] = 1

def convolve2(func):
    def wrapper(smartImage, B):
        img = smartImage.img
        hw = (int)((len(B) - 1) / 2)
        hh = (int)((len(B[0]) - 1) / 2)
        length, width = len(img), len(img[0])
        B = np.array(B)

        retVal = np.zeros([length, width])

        start = time.perf_counter()

        huh = [arr[i - 3: i + 3] for i, arr in enumerate(img)]
        print(huh)
        # for row in range(hh, length - hh):
        #     for pixel in range(hw, width - hw):
        #         result = func([arr[pixel-hh:pixel+hh+1]
        #                        for arr in img[row-hw:row+hw+1]], B)
        #         if result:
        #             retVal[row][pixel] = 1

        end = time.perf_counter()
        print(end - start)
        return SmartImage(retVal)
    return wrapper


def convolve(func):
    def wrapper(smartImage, B):
        img = smartImage.img
        hw = (int)((len(B) - 1) / 2)
        hh = (int)((len(B[0]) - 1) / 2)
        length, width = len(img), len(img[0])
        B = np.array(B)

        retVal = np.zeros([length, width])

        start = time.perf_counter()

        for row in range(hh, length - hh):
            for pixel in range(hw, width - hw):
                result = func([arr[pixel-hh:pixel+hh+1]
                               for arr in img[row-hw:row+hw+1]], B)
                if result:
                    retVal[row][pixel] = 1

        end = time.perf_counter()
        print(end - start)
        return SmartImage(retVal)
    return wrapper


class SmartImage:
    def __init__(self, img: np.ndarray):
        try:
            self.img = self.rgb2gray(img).img
        except:
            self.img = img

    def binarize(self):
        self.wiener()
        contrast = SmartImage(self.img).contrast_gradient()
        canny = SmartImage(self.img).canny()
        otsu = SmartImage(self.img).otsu().invert()
        (contrast & canny | otsu).invert().save()

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

    # dilation
    # @convolve
    @convolve2
    def __add__(smartImage, B):
        return np.mean(np.logical_and(smartImage, B)) > 0

        # erosion
    @convolve
    def __sub__(smartImage, B):
        return np.mean(np.logical_and(smartImage, B)) == 1

    def __and__(self, smartImage):
        return SmartImage(self.img & smartImage.img)

    def __or__(self, smartImage):
        return SmartImage(self.img | smartImage.img)

    def contrast_gradient(self, power=1, thresh=0.1):
        alpha = sqrt(np.var(self.img) / 128) ** power
        epsilon = 0.00001 * np.ones(self.img.shape)
        Max = maximum_filter(self.img, size=3)
        Min = minimum_filter(self.img, size=3)
        dif = (Max - Min)
        self.img = alpha * (dif / (Max + Min + epsilon)) + (1 - alpha) * dif
        self.img[self.img >= thresh] = 1
        self.img[self.img < thresh] = 0
        self.img = self.img.astype(int)
        return self

    def rgb2gray(self, img):
        self.img = rgb2gray(img)
        return self

    def otsu(self):
        self.img = (self.img > threshold_otsu(self.img)).astype(int)
        return self

    def canny(self):
        self.img = feature.canny(self.img).astype(int)
        return self

    def invert(self):
        self.img = (np.ones(self.img.shape) - self.img).astype(int)
        return self

    def save(self):
        imsave("../DIB-FrontEnd/src/assets/output/output.png",
               self.img, cmap='gray')
