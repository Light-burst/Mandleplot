import os
import numpy as np
from PIL import Image
import numba

ITERATIONS = 250
POWER = 6


@numba.njit(fastmath=True, parallel=True)
def escape_length(complex_plane):
    width, height, _ = complex_plane.shape
    result = np.zeros((width, height, 1), dtype=np.uint8)
    for b in numba.prange(-height//2, height//2):
        for a in range(-width//2, width//2):
            c = complex(a/(width//4), b/(height//4))
            z = np.power(complex(0, 0), POWER) + c
            for i in range(ITERATIONS):
                z = np.power(z, POWER) + c
                if np.absolute(z) > 2.0:
                    result[b][a] = i+1
                    break
            if np.absolute(z) < 2.0:
                result[b][a] = 0
    return result


def sigmoid_decay(x, shift=3, fade=1):
    t = (x/fade) - shift
    return 1 / (np.exp(t)+1)



def main():
    image = Image.open(os.path.join(os.getcwd(), "img4.jpg"), 'r')
    npimg = np.array(image)
    width = npimg.shape[0]
    height = npimg.shape[1]
    complex_plane = np.array([[[complex(b/(width//4), a/(height//4))]
                               for b in range(-width//2, width//2)]
                              for a in range(-height//2, height//2)], dtype=complex)
    mandlebrot = escape_length(complex_plane)
    mandlebrot_mask = np.where(mandlebrot == 0, True, False)
    mandleimg = np.where(mandlebrot_mask, 0, npimg)
    mandleimg = np.multiply(np.vectorize(sigmoid_decay)(mandlebrot), mandleimg)
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
