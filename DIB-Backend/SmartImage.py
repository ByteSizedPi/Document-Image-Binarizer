import math
import numpy as np
from numpy.fft import fft2, ifft2
from matplotlib.pyplot import imsave
from skimage import feature
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from skimage.restoration import denoise_wavelet, estimate_sigma
from skimage.morphology import skeletonize, closing
from scipy import ndimage
from scipy.ndimage import maximum_filter, minimum_filter, median_filter, generic_filter
from scipy.signal.windows import gaussian
from scipy.signal import convolve2d, wiener
from scipy.ndimage._morphology import distance_transform_edt
from matplotlib.image import imread

SIGMA = imread(
    'C:/Personal/Coding/Angular/DIB/DIB-FrontEnd/src/assets/DIBCO_2016/12.bmp')


def convolve(func):
    def wrapper(image):
        img = image.img

        length, width = len(img), len(img[0])
        B = np.array([[1] * 3] * 3)

        retVal = np.zeros([length, width])

        for row in range(1, length - 1):
            for pixel in range(1, width - 1):
                retVal[row][pixel] = func([arr[pixel-1:pixel+2]
                                           for arr in img[row-1:row+2]])
                # if result:
                #     retVal[row][pixel] = True
        return SmartImage(retVal)
    return wrapper


class SmartImage:
    def __init__(self, img: np.ndarray):
        try:
            self.img = self.rgb2gray(img).img
        except:
            self.img = img

    def mean(self, window=3):
        return SmartImage(convolve2d(self.img, np.array([[1] * window] * window) / 9, boundary='symm', mode='same'))

    def sigma(self):
        s = SmartImage(SIGMA)
        # s.img[s.img < 0.0001] = 0.0001
        return s

    @convolve
    def w(img):
        sigma = np.var(img)
        # mu = np.mean(img)
        return sigma

    def variance(self):
        return SmartImage(generic_filter(self.img, np.var, (3, 3)))

    def wiener1(self, kernel_size=3):
        return SmartImage(wiener(self.img, (kernel_size, kernel_size)))

    def binarize(self):
        # s = self.wiener1(10).contrast_gradient().normalize()
        # s.save('gradient')
        # s.img[s.img > 0.08] = 1
        # SmartImage(closing(skeletonize(s.img))).save('closing')
        # s = self.sigma()
        # s.save('s')
        self.realBin()

    def realBin(self):
        otsu = self.wiener1(3).save('1. wiener').otsu().save('2. otsu')
        stroke_width = (~otsu).avg_stroke_width()
        median = otsu.median_filter(stroke_width).save('5. median')
        (median | otsu).save()

        # binary operations
    def __and__(self, smartImage):
        return SmartImage(self.img & smartImage.img)

    def __or__(self, smartImage):
        return SmartImage(self.img | smartImage.img)

    def __sub__(self, smartImage):
        return SmartImage(self.img - smartImage.img)

    def __mul__(self, smartImage):
        return SmartImage(self.img * smartImage.img)

    def __truediv__(self, smartImage):
        return SmartImage(self.img / smartImage.img)

    def __add__(self, smartImage):
        return SmartImage(self.img + smartImage.img)

    def __invert__(self):
        dtype = self.img.dtype
        diff = (np.ones(self.img.shape) - self.img).astype(dtype)
        return SmartImage(diff)

    # denoising

    # kernel used for wiener filter
    def gaussian_kernel(kernel_size=3):
        # arbitrary selection of standard deviation of gaussian distribution
        stdev = kernel_size / 3
        # create 1D discrete gaussian array
        gauss_row = gaussian(M=kernel_size, std=stdev)
        # make row vector to column
        gauss_col = gauss_row.reshape(kernel_size, 1)
        # from 1D to 2D discrete gaussian array
        gauss2D = np.dot(gauss_col, gauss_col.transpose())
        # normalize 2D array
        gauss2D_normalized = gauss2D / np.sum(gauss2D)
        # return 2D kernel
        return gauss2D_normalized

    # adaptive wiener filter for nosie removal
    def wiener(self, kernel=gaussian_kernel(kernel_size=3), K=1000):
        # fast fourier transform from spatial domain to frequency domain
        kernel = fft2(kernel, s=self.img.shape)
        # core formula for wiener
        kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
        # fft of copy of image
        copy = fft2(np.copy(self.img))
        # calculate g(x) = f(x) * h(x)
        copy *= kernel
        # return to spatial domain
        return SmartImage(np.abs(ifft2(copy)))

    # median filter to remove spots
    def median_filter(self, size=3):
        return SmartImage(median_filter(self.img, size=size))

    # structural modifications

    # normalizing array values between 0 and 1: (list - min) / (max - min)
    def normalize(self):
        max_val = np.max(self.img)
        min_val = np.min(self.img)
        norm = (self.img - min_val) / (max_val - min_val)
        return SmartImage(norm)

    def contrast_gradient(self, power=1, thresh=0.1):
        alpha = math.sqrt(np.var(self.img) / 128) ** power
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
        return SmartImage(self.img > threshold_otsu(self.img))

    def canny(self):
        return SmartImage(feature.canny(self.img))

    def save(self, name='output'):
        imsave(f"../DIB-FrontEnd/src/assets/output/{name}.png",
               self.img, cmap='gray')
        return SmartImage(self.img)

    def avg_stroke_width(self):
        distances = distance_transform_edt(self.img)
        (~SmartImage(distances)).save('3. distance_transform')

        skeleton = skeletonize(self.img)
        (~SmartImage(skeleton)).save('4. skeletonized')

        d = distances[skeleton]

        stroke_width = np.mean(d) * 2
        rounded = math.floor(stroke_width) + 1
        return rounded
