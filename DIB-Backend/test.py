from Image import SmartImage
from images import getImages, showImages as show, graph
from statistics import mode

import numpy as np
from math import sqrt

(images, truths) = getImages()


def method(im):
    contrast = SmartImage(im).rgb2gray().contrast_gradient()
    otsu = SmartImage(contrast.img).otsu()
    canny = SmartImage(im).rgb2gray().canny()
    return SmartImage(otsu.img & canny.img).invert()


# modified = []
# for i in range(len(images)):
#     modified.append(method(images[i]))
#     images[i] = Image(images[i])
#     truths[i] = Image(truths[i])

og = SmartImage(images[0])
sobel0 = SmartImage(images[0]).sobel(0)
sobel1 = SmartImage(images[0]).sobel(1)
sobel2 = SmartImage(images[0]).sobelMag()

s = method(images[1])
# show([images[0], modified[0], truths[0], images[1], modified[1], truths[1]])
# show([og, s])

print("hello world")

# show(images)
# cur = img[0][0]
# arr = []

# for row in img:
#     for pixel in row:
#         arr.append(cur - pixel)
#         cur = pixel

# graph(arr)
# EW = mode(edgeArray(intersection))


# def extract(self, edgeWidth):
#     edgeWidth += not edgeWidth & 0b1
#     start = (edgeWidth - 1) / 2

#     def end(dim):
#         return len(dim) - start - 1

#     # for row in range(start, end(self.img)):
#     #     for pixel in range(start, end(self.img[0])):
#     #         mean = 0
#     #         for i in range(-start, start):
#     #             for j in range(-start, start):
#     #                 cur = self.img[row + i][pixel + j]
#     #                 mean += cur

#     mean = np.mean(self.img)
#     stdev = sqrt(np.var(self.img)) / 2
#     thresh = np.zeros(self.img.shape)
#     thresh[self.img <= mean + stdev] = 1
#     self.img = thresh
#     return self


# inverted = Image(intersection.img).invert()
# extraction = extract(intersection)

# print(widths)
# graph(edgeArray(gray.img))
# show([Image(img), contrast, otsu, canny, intersection])
