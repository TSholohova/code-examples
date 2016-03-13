import numpy as np


# v1_vector(x : 1d np.array of int)
# returns (values : 1d np.array of int, repeats : 1d np.array of int)
def v1_vector(x):
    jump = np.concatenate(([0], np.where(np.diff(x) != 0)[0] + 1, [len(x)]))
    return (x[jump[0:len(jump) - 1]], np.diff(jump))


# v2_non_vector(x : 1d np.array of int)
# returns (values : 1d np.array of int, repeats : 1d np.array of int)
def v2_non_vector(x):
    values = [x[0]]
    repeats = [1]
    for i in range(1, len(x)):
        if x[i] == values[-1]:
            repeats[-1] += 1
        else:
            values.append(x[i])
            repeats.append(1)
    return (values, repeats)


# v3_part_vector(x : 1d np.array of int)
# returns (values : 1d np.array of int, repeats : 1d np.array of int)
def v3_part_vector(x):
    jump = [0]
    for i in range(1, len(x)):
        if x[i] != x[i-1]:
            jump.append(i)
    return (x[jump], np.diff(jump + [len(x)]))


# gen(0 <= size <= 4 : int) returns tuple(x : 1d np.array of int, )
def gen(size):
    shape = np.array([100, 1000, 10000, 100000, 1000000])
    values = np.random.randint(0, 1000, shape[size])
    repeats = np.random.randint(0, 10, shape[size])
    return (np.repeat(values, repeats), )
