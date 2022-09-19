import sys
from images import getImages
from Image import SmartImage

index = (int)(sys.argv[1]) - 1
(original, ideal) = getImages()


def method(im):
    contrast = SmartImage(im).rgb2gray().contrast_gradient()
    # otsu = SmartImage(contrast.img).otsu()
    # canny = SmartImage(im).rgb2gray().canny()
    # return SmartImage(otsu.img & canny.img).invert()
    return contrast


# print(method(original[index]).img.shape)
# print(original[index].shape)
SmartImage(original[index]).save()
# method(original[index]).save()
