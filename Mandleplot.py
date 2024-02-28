import os
import numpy as np
from PIL import Image

ITERATIONS = 50
POWER = 6



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


def sigmoid_decay(x, shift, fade=1):
    t = (x/fade) - shift
    return 1 / (np.exp(t)+1)



def main():
    image = Image.open(os.path.join(os.getcwd(), "img.jpg"), 'r')
    npimg = np.array(image)
    k=0
    height = npimg.shape[0]
    width = npimg.shape[1]
    total = height*width
    for b in range(-height//2, height//2):
        for a in range(-width//2, width//2):
            z = complex(a/(width//4), b/(height//4))
            k += 1
            percent = 100*(k/total)
            if percent%1==0:
                print(percent)
            steps = escape_length(z)
            if steps<0:
                npimg[b+height//2][a+width//2] = 0
            elif steps!=0:
                pixel_value = npimg[b+height//2][a+width//2]
                # npimg[b+height//2][a+width//2] = (pixel_value/3.017) * (-np.arctan((steps-10)/2)+(np.pi/2))
                npimg[b + height // 2][a + width // 2] = pixel_value * sigmoid_decay(steps, 3, 1.5)
    image = Image.fromarray(npimg)
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
