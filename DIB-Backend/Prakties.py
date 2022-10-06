import numpy as np
from matplotlib.pyplot import imsave


def convolve(func):
    def wrapper(smartImage, B):
        img = smartImage.img
        hw = (int)((len(B) - 1) / 2)
        hh = (int)((len(B[0]) - 1) / 2)
        length, width = len(img), len(img[0])
        B = np.array(B)

        retVal = np.zeros([length, width])

        for row in range(hh, length - hh):
            for pixel in range(hw, width - hw):
                result = func([arr[pixel-hh:pixel+hh+1]
                               for arr in img[row-hw:row+hw+1]], B)
                if result:
                    retVal[row][pixel] = 1
        return SetOperation(retVal)
    return wrapper


class SetOperation:
    def __init__(self, img: np.ndarray):
        self.img = img.astype(bool)

    # dilation
    @convolve
    def __add__(smartImage, B):
        return np.mean(np.logical_and(smartImage, B)) > 0

    # erosion
    @convolve
    def __sub__(smartImage, B):
        return np.mean(smartImage & B) == 1

    # intersection
    def __and__(self, smartImage):
        return SetOperation(self.img & smartImage.img)

    # union
    def __or__(self, smartImage):
        return SetOperation(np.bitwise_or(self.img, smartImage.img))

    # complement
    def __invert__(self):
        return SetOperation(~self.img)

    def save(self):
        imsave("../DIB-FrontEnd/src/assets/output/output.png",
               self.img, cmap='gray')
