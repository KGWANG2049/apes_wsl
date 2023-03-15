import os
import random
import time
import numpy as np
from matplotlib import pyplot as plt

from path_gen import gmm_dist_generator
import torch.distributions as dist
import torch.nn.functional
import torch.optim as optim
from torch.optim import Adam
from collections import deque
from RRT_Connect_GMM import MyValidStateSampler
from tensorboardX import SummaryWriter
from experience import plan
from ran_rrtconnect import random_plan
from mp2d.scripts.planning import Planning
from mp2d.scripts.manipulator import manipulator
from apes.network_2d import APESCriticNet, APESGeneratorNet
from mp2d.scripts.utilities import load_planning_req_dataset
from torch.distributions.dirichlet import Dirichlet

BUFFER_MAX = 10
REPLAY_SAMPLE_SIZE = 2
TRAIN_T = 100
SAVE_INTERVAL = 50
start = time.time()
recent_steps = []
TARGET_ENTROPY = [-150.0]
LOG_ALPHA_INIT = [-20.0]
LR = 1e-4
LOG_ALPHA_MIN = -10.
LOG_ALPHA_MAX = 20.
cwd = os.getcwd()
dof = 2
links = [0.5, 0.5]
ma = manipulator(dof, links)
pl = Planning(ma)
pl_req_file_name = "/home/wangkaige/Project/apes/easy_pl_req_250_nodes.json"
planning_requests = load_planning_req_dataset(pl_req_file_name)
replay_buffer = deque(maxlen=BUFFER_MAX)
writer = SummaryWriter()
if __name__ == '__main__':

    SV = np.array(2)
    GV = np.array(2)
    W = np.array(50)
    data = np.array([])
    OC = np.array(np.shape(pl.get_occupancy_map(planning_requests[1])))
    device = torch.device("cpu")
    gen_model = APESGeneratorNet().float().to(device)
    gen_model.eval()
    critic_model = APESCriticNet().float().to(device)
    critic_model.eval()
    gen_model_optimizer: Adam = optim.Adam(gen_model.parameters(), lr=LR)
    critic_model_optimizer: Adam = optim.Adam(critic_model.parameters(), lr=LR)
    log_alpha = torch.tensor(LOG_ALPHA_INIT, requires_grad=True, device=device)
    alpha_optim = optim.Adam([log_alpha], lr=LR)
    torch.autograd.set_detect_anomaly(True)
    writer = SummaryWriter("Loss_Function")
    critic_losses = []
    gen_losses = []
    alpha_losses = []
    for p in range(5):
        for i in range(0, BUFFER_MAX):
            # ran_idx = torch.randint(low=0, high=4000, size=(1,))
            pl_req = planning_requests[i]
            OC = pl.get_occupancy_map(pl_req)
            OC = OC.reshape([1, OC.shape[0], -1])
            print("oc size:", OC.shape)
            SV = pl_req.start
            GV = pl_req.goal
            pl.generate_graph_halton(150)
            pr = pl.search(pl_req)
            # VALUE_ESTIMATE = pr.checked_counts
            # VALUE_ESTIMATE = torch.tensor(VALUE_ESTIMATE)
            OC = torch.tensor(OC).to(device)
            SV = torch.tensor(SV).to(device)
            GV = torch.tensor(GV).to(device)
            diri_dist = Dirichlet(gen_model(OC, SV, GV))
            W = diri_dist.sample()

            # print("W", W, W.shape)
            gmm_dist = gmm_dist_generator(W)
            VALUE_ESTIMATE = plan(pl_req, gmm_dist)[0]
            VALUE_ESTIMATE = torch.tensor(VALUE_ESTIMATE).to(device)
            # print("GMM maxcount", VALUE_ESTIMATE)
            experience = ([OC, SV, GV, W, VALUE_ESTIMATE])
            replay_buffer.append(experience)
            VALUE_RAN = random_plan(pl_req)
            # VALUE_RAN_SUM = VALUE_RAN + VALUE_RAN_SUM
            # VALUE_ESTIMATE_SUM = VALUE_ESTIMATE.cpu() + VALUE_ESTIMATE_SUM
            # del MVS, RVS
            print('Waiting for buffer size ... {}/{}'.format(len(replay_buffer), BUFFER_MAX))
        for i in range(5):

            # labels = sampled_oc, sampled_start_v, sampled_goal_v, sampled_coefficients, sampled_values
            sampled_evaluations = random.sample(replay_buffer, REPLAY_SAMPLE_SIZE)
            # print("sssssss", sampled_evaluations)
            # data_labels = {sampled_evaluations[i]: labels[i] for i in range(len(sampled_evaluations))}
            sampled_oc = torch.stack([t[0] for t in sampled_evaluations])
            # print("OC:", sampled_oc, sampled_oc.shape)
            sampled_start_v = torch.stack([t[1] for t in sampled_evaluations])
            # print("start_v:", sampled_start_v, sampled_start_v.shape)
            sampled_goal_v = torch.stack([t[2] for t in sampled_evaluations])
            # print("goal_v:", sampled_goal_v, sampled_goal_v.shape)
            sampled_coefficients = torch.stack([t[3] for t in sampled_evaluations])
            # print("sampled_coefficient:", sampled_coefficients, sampled_coefficients.shape)
            sampled_values = torch.stack([t[4] for t in sampled_evaluations])
            # print("value:", sampled_values, sampled_values.shape)

            critic_loss = 0
            for j in range(REPLAY_SAMPLE_SIZE):
                # cri_idx = j + (i * REPLAY_SAMPLE_SIZE)
                mean, std = critic_model(sampled_oc[j], sampled_start_v[j], sampled_goal_v[j],
                                         sampled_coefficients[j])
                std = torch.exp(std)
                print("mean:", mean, "std:", std)
                priori_pro = dist.Normal(mean, std)
                # print("posterior:", priori_pro)
                posterior_prob = priori_pro.log_prob(sampled_values[j])
                print("posterior_prob:", posterior_prob, posterior_prob.shape)
                # print("test", priori_pro.log_prob(a).exp())
                critic_loss = critic_loss + (-posterior_prob) / REPLAY_SAMPLE_SIZE

            critic_model_optimizer.zero_grad()

            critic_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(critic_model.parameters(), 1.0)
            critic_model_optimizer.step()
            writer.add_scalar('Critic Loss', critic_loss.item(), global_step=p * 5 + i)

            # Update generator
            gen_objective = 0
            for k in range(REPLAY_SAMPLE_SIZE):
                sampled_coefficients, entropy = gen_model.rsample(sampled_oc[k], sampled_start_v[k], sampled_goal_v[k])
                mean, std = critic_model(sampled_oc[k], sampled_start_v[k], sampled_goal_v[k], sampled_coefficients)

                # print("entropy", entropy)
                dual_terms = (log_alpha.exp().detach() * entropy)
                # print("dual_term", dual_terms)
                gen_objective = (gen_objective + mean - dual_terms) / REPLAY_SAMPLE_SIZE

                # gen_objective = (gen_objective + mean) / REPLAY_SAMPLE_SIZE

            gen_model_optimizer.zero_grad()
            gen_objective.backward()
            torch.nn.utils.clip_grad_norm_(gen_model.parameters(), 1.0)
            gen_model_optimizer.step()
            writer.add_scalar('Generator Objective', gen_objective.item(), global_step=p * 5 + i)
            print("gen_objective", gen_objective)

            # update loss
            alpha_loss = 0
            for m in range(REPLAY_SAMPLE_SIZE):
                # alp_idx = m + REPLAY_SAMPLE_SIZE * i
                # dir_dist = Dirichlet(sampled_coefficients[j])
                sampled_coefficients, entropy = gen_model.rsample(sampled_oc[m], sampled_start_v[m], sampled_goal_v[m])

                print("entropy", entropy)
                alpha_loss_single = log_alpha.exp() * ((entropy - torch.tensor(
                    TARGET_ENTROPY, device=device, dtype=torch.float32)).detach())
                alpha_loss = (alpha_loss + alpha_loss_single) / REPLAY_SAMPLE_SIZE
            critic_model_optimizer.zero_grad()
            gen_model_optimizer.zero_grad()
            alpha_optim.zero_grad()
            alpha_loss.backward()
            with torch.no_grad():
                log_alpha.grad *= (((-log_alpha.grad >= 0) | (log_alpha >= LOG_ALPHA_MIN)) &
                                   ((-log_alpha.grad < 0) | (log_alpha <= LOG_ALPHA_MAX))).float()  # ppo
            alpha_optim.step()
            writer.add_scalar('Alpha Loss', alpha_loss.item(), global_step=p * 5 + i)

    writer.close()






    
    
    # tensorboard --logdir=/home/wangkaige/Project/apes/LOSS_FUNCTION """
