from skimage.morphology import closing
from images import getImages

im = getImages()[0]

im.binarize()
