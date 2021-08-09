import math
import numpy as np
from numpy import unravel_index


def sign(val):
    if val < 0:
        return -1
    return 1


a = np.array([[3, 2, 4], [2, 0, 2], [4, 2, 3]],np.dtype(float))

for k in range(10):
    a_abs = np.array([abs(num) for num in a.flatten()])
    np.reshape(a_abs,(3,3))
    maxindex = a_abs.argmax()
    max_ind = (unravel_index(a_abs.argmax(), a.shape))
    print(max_ind)
    i = max_ind[0]
    j = max_ind[1]
    theta = (a[j][j] - a[i][i]) / 2 * a[i][j]
    t = sign(theta) / (abs(theta) + math.sqrt(theta ** 2 + 1))
    c = 1 / math.sqrt(t ** 2 + 1)
    s = t * c

    print("t = ", t)
    print("c = ", c)
    print("s = ", s)
    for r in range(3):
        if r != i and r != j:
            a_ri = c * a[r][i] - s * a[r][j]
            # print("a_ri = ",a_ri)
            a_rj = c * a[r][j] + s * a[r][i]
            # print("a_rj = ", a_rj)
            a[r][i] = a_ri
            a[i][r] = a_ri
            a[r][j] = a_rj
            a[j][r] = a_rj
            # print("a[r][j] = ",a[r][j])

    a_ii = c ** 2 * a[i][i] + s ** 2 * a[j][j] - 2 * c * s * a[i][j]
    # print("a_ii = ", a_ii)
    a_jj = s ** 2 * a[i][i] + c ** 2 * a[j][j] + 2 * c * s * a[i][j]
    # print("a_jj = ", a_jj)
    a[i][i] = a_ii
    a[j][j] = a_jj
    # print("a[j][j] = ", a[j][j])
    a[i][j] = 0
    a[j][i] = 0

    print(a)
