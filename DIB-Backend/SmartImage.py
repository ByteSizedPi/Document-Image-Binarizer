import math
import numpy as np
from math import sqrt
from numpy.fft import fft2, ifft2
from matplotlib.pyplot import imsave
from skimage import feature
from skimage.filters import threshold_sauvola, threshold_otsu, threshold_mean
from skimage.color import rgb2gray, gray2rgb
from skimage.draw import disk
from skimage.restoration import denoise_wavelet
from skimage.morphology import skeletonize
from skimage.filters.rank import noise_filter
from skimage.morphology import disk
from skimage.morphology import disk
from scipy import ndimage
from scipy.ndimage import maximum_filter, minimum_filter, gaussian_filter, sobel, median_filter
from scipy.signal import convolve2d, gaussian
from scipy.ndimage._morphology import distance_transform_edt
from scipy.ndimage._morphology import binary_closing, binary_opening


class SmartImage:
    def __init__(self, img: np.ndarray):
        try:
            self.img = self.rgb2gray(img).img
        except:
            self.img = img

    def binarize(self):
        im = SmartImage(self.img).normalize()
        denoised = im.denoise().wiener().normalize()
        denoised.save('1. denonised')

        otsu = denoised.otsu()
        otsu.save('2. otsu_thresholded')
        stroke_width = (~otsu).avg_stroke_width()

        median = otsu.median_filter(stroke_width)
        median.save('5. median filter')

        (~((~median) & (~otsu))).save('6. result')

    def normalize(self):
        max_val = np.max(self.img)
        min_val = np.min(self.img)
        norm = (self.img - min_val) / (max_val - min_val)
        return SmartImage(norm)

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
        return SmartImage(np.abs(ifft2(copy)))

    def __and__(self, smartImage):
        return SmartImage(np.logical_and(self.img, smartImage.img))

    def __or__(self, smartImage):
        return SmartImage(self.img | smartImage.img)

    def __sub__(self, smartImage):
        return SmartImage(self.img - smartImage.img)

    def __invert__(self):
        return SmartImage(np.ones(self.img.shape) - self.img)

    def contrast_gradient(self, power=1, thresh=0.1):
        alpha = sqrt(np.var(self.img) / 128) ** power
        epsilon = 0.00001 * np.ones(self.img.shape)
        Max = maximum_filter(self.img, size=3)
        Min = minimum_filter(self.img, size=3)
        dif = (Max - Min)
        retVal = alpha * (dif / (Max + Min + epsilon)) + (1 - alpha) * dif
        retVal[retVal >= thresh] = 1
        retVal[retVal < thresh] = 0
        retVal = retVal.astype(int)
        return SmartImage(retVal)

    def rgb2gray(self, img):
        return SmartImage(rgb2gray(img))

    def otsu(self):
        return SmartImage((self.img > threshold_otsu(self.img)).astype(int))

    def canny(self):
        return SmartImage(feature.canny(self.img))

    def save(self, name='output'):
        imsave(f"../DIB-FrontEnd/src/assets/output/{name}.png",
               self.img, cmap='gray')

    def avg_stroke_width(self):
        distances = distance_transform_edt(self.img)
        SmartImage(distances).save('3. distance_transform')

        skeleton = skeletonize(self.img)
        SmartImage(skeleton).save('4. skeletonized')

        d = distances[skeleton]

        stroke_width = np.mean(d) * 2
        rounded = math.floor(stroke_width) + 1
        return rounded

    def median_filter(self, size=3):
        return SmartImage(median_filter(self.img, size=size))

    def denoise(self):
        return SmartImage(denoise_wavelet(self.img))
