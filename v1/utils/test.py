import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print(a)

b = np.zeros(a.shape)
b[:, 0] = np.sin(a[:, 0])
b[:, 1] = np.sin(a[:, 1])
b[:, 2] = np.sin(a[:, 2])

print(b)
print("\n")
print(np.transpose(a[:, 0]), "\n")
c = np.concatenate((np.transpose(np.cos(a[:, 0])), np.transpose(np.sin(a[:, 0])), np.transpose(a[:, 0])), axis=1)
print(c)