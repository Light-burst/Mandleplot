import os
import numpy as np
from PIL import Image

ITERATIONS = 500
POWER = 2



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


def sigmoid_decay(x, shift=20, fade=1.2):
    t = (x/fade) - shift
    return 1 / (np.exp(t)+1)



def main():
    # image = Image.open(os.path.join(os.getcwd(), "img.jpg"), 'r')
    # npimg = np.array(image)
    npimg = np.zeros((12000, 12000, 3), dtype=np.uint8)
    npimg = npimg + 255
    height, width, _ = npimg.shape
    complex_plane = np.array([[[complex(b/(width//4), a/(height//4))]
                               for b in range(-width//2, width//2)]
                              for a in range(-height//2, 0)], dtype=complex)

    mandlebrot = np.zeros((height//2, width, 1), dtype=np.uint8)
    undone = np.ones((height//2, width, 1), dtype=bool)

    Zn = np.zeros_like(complex_plane)
    for iteration in range(ITERATIONS):
        Zn = np.power(Zn, 2) + complex_plane
        escaped = np.absolute(Zn) > 2
        mandlemask = np.logical_and(escaped, undone)
        mandlebrot = np.where(mandlemask, iteration+1, mandlebrot)
        undone = np.logical_and(undone, np.logical_not(mandlemask))

    mandlebrot = np.pad(mandlebrot, ((0, height//2), (0, 0), (0, 0)), "reflect")
    mandlebrot_mask = np.where(mandlebrot==0, True, False)
    npimg = np.where(mandlebrot_mask, 0, npimg)
    mandleimg = np.multiply(np.vectorize(sigmoid_decay)(mandlebrot), npimg)
    # mandleimg = np.multiply((mandlebrot), npimg)
    image = Image.fromarray(mandleimg.astype(np.uint8))
    image.show()
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
