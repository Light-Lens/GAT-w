import numpy

x = [1, 2, 3, 4]
w1 = [0.2, 0.5, 0.3, 0.8]
b1 = 0.5

o1 = numpy.dot(x, w1) + b1

w2 = [0.4, 0.2, 0.3, 0.9]
b2 = 0.2

o2 = numpy.dot(o1, w2) + b2

w3 = [0.6, 0.2, 0.3, 0.5]
b3 = 0.7

o3 = numpy.dot(o2, w3) + b3

print(o3)
