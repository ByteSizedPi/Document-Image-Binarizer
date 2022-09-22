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

def convolve(func):
    def wrapper(smartImage, B):
        img = smartImage.img
        length = len(img)
        width = len(img[0])
        hw = (int)((len(B) - 1) / 2)
        hh = (int)((len(B[0]) - 1) / 2)
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
            self.img = rgb2gray(img)
        except:
            self.img = img

    def binarize(self):
        contrast = SmartImage(self.img).contrast_gradient()
        canny = SmartImage(self.img).canny()
        (canny.img & contrast.img).save()
        # contrast = self.contrast_gradient()
        # otsu = self.otsu().save()
        # canny = self.canny().save()

    # dilation
    @convolve
    def __add__(smartImage, B):
        return np.mean(np.logical_and(smartImage, B)) > 0

    # erosion
    @convolve
    def __sub__(smartImage, B):
        return np.mean(np.logical_and(smartImage, B)) == 1

    def contrast_gradient(self, power=1):
        alpha = sqrt(np.var(self.img) / 128) ** power
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

    def save(self):
        imsave("../DIB-FrontEnd/src/assets/output/output.png",
               self.img, cmap='gray')
