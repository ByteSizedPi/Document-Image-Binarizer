from images import getImages
import numpy as np
from matplotlib import pyplot as plt
from collections import Counter

images = getImages()
im = images[0].binarize()
flat = [el for arr in im.img for el in arr]
# flat = [i for i in flat if i > 0]
plt.hist(flat, range=(0, 16), bins=15)
print(Counter(flat).most_common(1))
plt.show()
