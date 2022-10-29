from SmartImage import SmartImage
from matplotlib.image import imread


def read(path: str):
    return SmartImage(imread('C:/Personal/Coding/Angular/DIB/DIB-FrontEnd/src/assets/Input/' + path))


def getImages():
    return [read(str(i) + '.png') for i in range(1, 17)]
