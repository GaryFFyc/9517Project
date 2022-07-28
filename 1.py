import numpy as np

a = np.array([[1, 2], [3, 4], [5, 6]])
a1 = np.repeat(a, 3, axis=0)
print(a1)
a2 = np.tile(a, [3, 1])
print(a2)
a3 = a1 - a2
print(a3)
a4 = np.power(a3, 2)
print(a4)
a5 = np.sqrt(np.sum(a4, axis=1))
a5 = a5.reshape(3, 3)
print(a5)
mask1 = np.full((3, 3), 5)
print(mask1)
print(np.sum(np.sum(a5 < mask1, 1) == 3))


