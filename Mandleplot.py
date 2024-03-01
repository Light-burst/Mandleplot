import os
import numpy as np
from PIL import Image

ITERATIONS = 1250
POWER = 6
IMAGE_FILE = "img4.jpg"
BLANK_SIZE = 10000
SMOOTHING = True
FADE = 3
SHIFT = 3.7
MIRROR_INPUT = True



def escape_length(z: complex):
    init_z = z
    z = np.power(complex(0, 0), POWER) + init_z
    i = ITERATIONS
    while np.absolute(z) < 2.0 and i!=0:
        z = np.power(z, POWER) + init_z
        i -= 1
    if np.absolute(z) < 2.0:
        return -1
    else:
        return ITERATIONS - i


def sigmoid_decay(x, shift=SHIFT, fade=FADE):
    t = (x/fade) - shift
    return 1 / (np.exp(t)+1)



def main():
    if IMAGE_FILE != "":
        image = Image.open(os.path.join(os.getcwd(), IMAGE_FILE), 'r')
        npimg = np.array(image)
    else:
        npimg = np.zeros((BLANK_SIZE, BLANK_SIZE, 3), dtype=np.uint8)
        npimg = npimg + 255
    height, width, _ = npimg.shape
    if MIRROR_INPUT:
        npimg = npimg[:][0:height//2]
        npimg = np.pad(npimg, ((0, height // 2), (0, 0), (0, 0)), "reflect")
    complex_plane = np.array([[[complex(b/(width//4), a/(height//4))]
                               for b in range(-width//2, width//2)]
                              for a in range(-height//2, 0)], dtype=complex)

    mandlebrot = np.zeros((height//2, width, 1), dtype=np.uint8)
    undone = np.ones((height//2, width, 1), dtype=bool)

    Zn = np.zeros_like(complex_plane)
    for iteration in range(ITERATIONS):
        Zn = np.power(Zn, POWER) + complex_plane
        escaped = np.absolute(Zn) > 2
        mandlemask = np.logical_and(escaped, undone)
        mandlebrot = np.where(mandlemask, iteration+1, mandlebrot)
        undone = np.logical_and(undone, np.logical_not(mandlemask))
        progress = (100*iteration) / ITERATIONS
        if progress % 1 == 0:
            print(str(int(progress))+'%')

    mandlebrot = np.pad(mandlebrot, ((0, height//2), (0, 0), (0, 0)), "reflect")
    mandlebrot_mask = np.where(mandlebrot==0, True, False)
    mandleimg = np.where(mandlebrot_mask, 0, npimg)
    if SMOOTHING:
        mandleimg = np.multiply(np.vectorize(sigmoid_decay)(mandlebrot), mandleimg)
    image = Image.fromarray(mandleimg.astype(np.uint8))
    outpath = os.path.join(os.getcwd(), "Mandled_"+IMAGE_FILE)
    outpath = outpath if IMAGE_FILE != "" else outpath[:-1]+".jpg"
    image.save(outpath)
main()




# class Z:
#     def __init__(self, real, imaginary):
#         self.re = real
#         self.im = imaginary
#     __slots__ = ['re', 'im']
#
#     def __str__(self):
#         if self.re==0 and self.im==0:
#             return '0'
#         result = ""
#         if self.re != 0:
#             result += str(self.re)
#         if self.im == 1:
#             result += '+i'
#         elif self.im == -1:
#             result += "-i"
#         elif self.im != 0:
#             if self.im > 0 and self.re != 0:
#                 result += '+'
#             result += str(self.im)+'i'
#         return result
#     def __add__(self, other):
#         self.re += other.re
#         self.im += other.im
#         return self
#     def __sub__(self, other):
#         self.re -= other.re
#         self.im -= other.im
#         return self
#     def __mul__(self, other):
#         return Z(self.re*other.re - self.im*other.im, self.re*other.im + self.im*other.re)
#     def __pow__(self, power, modulo=None):
#         if power == 0:
#             return Z(0, 0)
#         if power == 1:
#             return self
#         sq = self**(power//2)
#         if power % 2 == 0:
#             return sq*sq
#         return self*(sq*sq)
#     def __eq__(self, other):
#         return self.re == other.re and self.im == other.im
#     def __ne__(self, other):
#         return not __eq__(self, other)
#
#     def length(self):
#         return math.sqrt(self.re**2 + self.im**2)
