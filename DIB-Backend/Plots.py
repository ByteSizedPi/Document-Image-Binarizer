from images import getImages
import numpy as np
from matplotlib import pyplot as plt
from collections import Counter

images = getImages()
im = images[11]
flat = [el for arr in im.img for el in arr]
flat = [i for i in flat if i > 0.01]
# plt.hist(flat, range=(0, 1), bins=50)
print(Counter(flat).most_common(3))
# plt.show()
