import torch
from copy import deepcopy
from mp2d.scripts.planning import Planning
from APES_2D_train import planning_requests
from mp2d.scripts.manipulator import *
import numpy as np

dof = 2
SV = np.array(2)
GV = np.array(2)
W = np.array(50)
links = [0.5, 0.5]
ma = manipulator(dof, links)
pl = Planning(ma)
planning_range_max = np.array([np.pi, np.pi])
planning_range_min = np.array([-np.pi, -np.pi])
pl_env = Planning(ma, planning_range_max, planning_range_min, resolution=0.05)
nodes_recording = {}
planning_requests_wo_obstacles = deepcopy(planning_requests)
# Find solutions in parallel using RRT
easy_path = "/home/wangkaige/Project/apes/easy_pl_req_250_nodes.json"
dataset = load_planning_req_dataset(easy_path)
# Define the weights for each multivariate Gaussian
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("...")
pl_req = planning_requests[98]
SV = pl_req.start
SV = torch.tensor(SV)
GV = pl_req.goal
GV = torch.tensor(GV)
req = pl_req
OC = pl.get_occupancy_map(req)
OC = torch.tensor(OC)
OC = OC.reshape([1, OC.shape[0], -1])
Gen_net = torch.load("/home/wangkaige/Project/apes/net.generator")
W = Gen_net(OC, SV, GV)
print("WEIGHT", W)
length_W = W.shape[1]
W = torch.tensor(W)
W = W.squeeze()
print(W)
_, ax = plt.subplots(1, 1)
x = torch.arange(0, 50)
ax.scatter(x, W.numpy())
y = np.zeros(50) + 0.02
plt.plot(x, y)
plt.show()
