from DIBArray import DIBArray
from matplotlib.image import imread


def read(path: str):
    return DIBArray(imread('C:/Personal/Coding/Angular/DIB/DIB-FrontEnd/src/assets/DIBCO_2016/' + path))


def getImages():
    return [read(str(i) + '.bmp') for i in range(1, 12)]
