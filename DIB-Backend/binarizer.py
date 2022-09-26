import sys
from images import getImages


index = (int)(sys.argv[1]) - 1
im = getImages()[index]

dim = 17
B = [[1] * dim] * dim

# im.binarize()


def dilate(): return im + B
def erode(): return im - B
def open(): return (im - B) + B
def close(): return (im + B) - B


(im + B).save()
