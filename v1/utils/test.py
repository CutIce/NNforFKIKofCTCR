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

print(a[0])
print(a[1])
# c = np.concatenate((np.transpose(np.cos(a[:, 0])), np.transpose(np.sin(a[:, 0])), np.transpose(a[:, 0])), axis=1)
# print(c)
print(sum(a)/len(a))

import torch

# c1 = torch.FloatTensor(np.array([1, 2, 3, 4, 5]), requires_grad=True)
# c2 = torch.FloatTensor(np.array([2, 3, 4, 5, 6]), requires_grad=True)
#
# criterion = torch.nn.MSELoss(reduce="mean")
#
# loss = criterion(c1, c2)


# print(loss)


print("------------------------------")

L1 = 210
L2 = 165
L3 = 110

Mb = np.array([
    [-L1, 0,     0],
    [-L1, L1-L2, 0],
    [-L1, L1-L2, L2-L3]
])


print(Mb)
T = np.zeros((4, 4))
T[0:3, 0:3] = 1/2 * Mb
print(1/2 * np.dot(Mb, np.ones((3, 1))))
T[0:3, 3:] = 1/2 * np.dot(Mb, np.ones((3, 1)))
T[3, 3] = 1


print(np.linalg.inv(T))

