from numpy import ndarray
import matplotlib.image as mpimg
from DIBArray import DIBArray


def read(path: str) -> DIBArray:
    return DIBArray(mpimg.imread('C:/Personal/Coding/Angular/DIB/DIB-FrontEnd/src/assets/DIBCO_2016/' + path))


def getImages() -> tuple[list[DIBArray], list[DIBArray]]:
    original: list[DIBArray] = []
    ideal: list[DIBArray] = []
    for i in range(1, 12):
        original.append(read(str(i) + '.bmp'))
        ideal.append(read(str(i) + '_gt.bmp'))
    return original, ideal
