from skimage.draw import disk
import sys
from images import getImages
from SmartImage import SmartImage
import numpy as np
from SetOperation import SetOperation


index = (int)(sys.argv[1]) - 1
im = getImages()[index]


# def method(im):
#     contrast = SmartImage(im).contrast_gradient()
#     otsu = SmartImage(contrast.img).otsu()
#     canny = SmartImage(im).canny()
#     return SmartImage(otsu.img & canny.img).invert()


dim = 11
B = [[1] * dim for j in range(dim)]

smartImage = SmartImage(im.img)
smartImage.binarize()


# dilated = smartImage - B
# dilated.save()
