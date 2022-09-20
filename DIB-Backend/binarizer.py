from skimage.draw import disk
import sys
from images import getImages
from SmartImage import SmartImage
import numpy as np
from SetOperation import SetOperation


index = (int)(sys.argv[1]) - 1
original, ideal = getImages()
im = ideal[index]

# def method(im):
#     contrast = SmartImage(im).rgb2gray().contrast_gradient()
#     otsu = SmartImage(contrast.img).otsu()
#     canny = SmartImage(im).rgb2gray().canny()
#     return SmartImage(otsu.img & canny.img)


dim = 17
B = [[1 for i in range(dim)] for j in range(dim)]


def dilate(): return im + B
def erode(): return im - B


def open(): return (im - B) + B
def close(): return (im + B) - B


img = erode()

img.save()
