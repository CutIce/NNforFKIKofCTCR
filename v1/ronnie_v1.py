import torch.nn as nn
from torch.nn import init

from matplotlib import pyplot as plt

from dataloader import *

# the structure of dataset points
# 6 absolute joint values
# 6 relative joint values
# pose of the base
# pose of the proximal sensor attached to the outermost tube
# pose of the sensor attached to the middle tube
# pose of the distal sensor attached to the most inner tube
# 6 + 6 + 7 + 7 + 7 + 7

df = pd.read_csv("../dataset/CRL-Dataset-CTCR-Pose.csv", header=None)
dt = df.values

train_data = np.zeros((dt.shape[0], 18))
for i in range(0, 12, 2):
    train_data[:, 3*(i//2)] = np.cos(dt[:, i])
    train_data[:, 3*(i//2)+1] = np.sin(dt[:, i])
    train_data[:, 3*(i//2)+2] = dt[:, i+1]

label_x_pose = dt[:, -7:]
label_tip_position = dt[:, -7:-4]
label_three_points = np.concatenate((dt[:, 12:15], dt[:, 19:22], dt[:, -7:-4]), axis=1)

print("All Dataset Shape: ", dt.shape)
print("Train data Shape: ", train_data.shape)
print("Label (x pose) Shape: ", label_x_pose.shape, "Label (tip position) Shape: ", label_tip_position.shape, "Label (3 points) Shape: ", label_three_points.shape)

seed = 10
np.random.seed(seed)

shuffle_idx = np.random.permutation(np.arange(dt.shape[0]))
train_data = train_data[shuffle_idx, :]
label_x_pose = label_x_pose[shuffle_idx, :]
label_tip_position = label_tip_position[shuffle_idx, :]
label_three_points = label_three_points[shuffle_idx, :]

ratio = 0.1
batch_size = 128

train_set = torch.tensor(train_data[:90000, :])
train_label_x_pose_set = torch.tensor(label_x_pose[:90000, :])

valid_set = torch.tensor(train_data[90000:, :])
valid_label_s_pose_set = torch.tensor(label_x_pose[90000:, :])

print(train_set.shape, train_label_x_pose_set.shape)
print(valid_set.shape, valid_label_s_pose_set.shape)

train_dataset = Data.TensorDataset(train_set, train_label_x_pose_set)
valid_dataset = Data.TensorDataset(valid_set, valid_label_s_pose_set)


class MLP(nn.Module):
    
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.hidden1 = nn.Linear(18, 256)
        self.a1 = nn.ReLU()

        self.hidden2 = nn.Linear(256, 1024)
        self.a2 = nn.ReLU()

        self.output = nn.Linear(1024, 7)

    def forward(self, x):
        y = self.hidden1(x)
        y = self.a1(y)

        y = self.hidden2(y)
        y = self.a2(y)

        return self.output(y)

net = MLP()
for params in net.parameters():
    init.normal_(params, mean=0, std=0.01)
print(net)



def train(net, train_dataset, valid_dataset, num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    loss = torch.nn.MSELoss()
    train_dataloader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = Data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    net = net.float()
    for epoch in range(num_epochs):
        total_loss = 0
        total_size = 0
        for i, (x, y) in enumerate(train_dataloader):
            pred = net(x.float())
            l = loss(pred, y.float())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            total_loss += l.detach()
            total_size += x.shape[0]
        train_ls.append(total_loss / total_size)

    plt.plot(list(range(len(train_ls))), train_ls)
    plt.show()



train(net, train_dataset, valid_dataset, 10, 0.001, 0.01, 1024)
