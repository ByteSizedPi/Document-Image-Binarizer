from typing import Tuple
from numpy import ndarray
import matplotlib.image as mpimg
from Image import SmartImage


def read(path: str):
    return mpimg.imread('C:/Personal/Coding/Angular/DIB/DIB-FrontEnd/src/assets/DIBCO_2016/' + path)


def getImages() -> Tuple[list[ndarray], list[ndarray]]:
    original: list[ndarray] = []
    ideal: list[ndarray] = []
    for i in range(1, 11):
        original.append(read(str(i) + '.bmp'))
        ideal.append(read(str(i) + '_gt.bmp'))
    return (original, ideal)
