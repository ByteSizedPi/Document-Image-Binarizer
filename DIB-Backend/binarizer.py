import sys
from images import getImages
index = (int)(sys.argv[1]) - 1
getImages()[index].binarize()
