from numpy import diff
from matplotlib import pyplot as plt
import sys
from images import getImages
import numpy as np

index = (int)(sys.argv[1]) - 1
im = getImages()[index]

im.binarize()
