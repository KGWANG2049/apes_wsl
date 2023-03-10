from mp2d.scripts.utilities import *
from mp2d.scripts.rrt_connect import RRTConnect
import matplotlib.pyplot as plt
import numpy as np
from time import sleep
from math import fabs
from path_gen import *
import torch
from mp2d.scripts.manipulator import *
from mp2d.scripts.planning import *

try:
    from ompl import util as ou
    from ompl import base as ob
    from ompl import geometric as og
except ImportError:
    # if the ompl module is not in the PYTHONPATH assume it is installed in a
    # subdirectory of the parent directory called "py-bindings."
    from os.path import abspath, dirname, join
    import sys

    sys.path.insert(0, join(dirname(dirname(abspath(__file__))), '/home/wangkaige/Project/mp2d/ompl-1.5.2/py-bindings'))
    from ompl import util as ou
    from ompl import base as ob
    from ompl import geometric as og

links = [0.5, 0.5]
dof = 2
ma = manipulator(dof, links)
pl = Planning(ma)
planning_range_max = np.array([np.pi, np.pi])
planning_range_min = np.array([-np.pi, -np.pi])
pl_env = Planning(ma, planning_range_max, planning_range_min, resolution=0.05)


class MyValidStateSampler(ob.ValidStateSampler):

    def __init__(self, gmm_dist, pl_req):
        self.si = self.get_si()
        super(MyValidStateSampler, self).__init__(self.si)
        self.name_ = "aaa"
        self.gmm_dist = gmm_dist
        self.pl_req = pl_req
        self.count = 0
        self.max_count = 0

    def sample(self, state):
        p = self.gmm_dist.sample()
        p = p.cpu().numpy()
        state[0] = p[0]
        state[1] = p[1]
        self.count += 1
        print(self.count)
        if self.count > self.max_count:
            self.max_count = self.count

        return True

    def get_count_max(self):
        return self.max_count

    def get_si(self):
        space = ob.RealVectorStateSpace(2)
        bounds = ob.RealVectorBounds(2)
        bounds.setLow(-4)
        bounds.setHigh(4)
        space.setBounds(bounds)
        ss = og.SimpleSetup(space)
        self.si = ss.getSpaceInformation()
        return self.si

    # def reset_counts(self):
    # self.count = 0
    def __call__(self, _, __):
        return self

    def isStateValid(self, state):
        obstacles = self.pl_req.obstacles
        is_valid = pl_env.manipulator.check_validity(state, obstacles)
        return is_valid

    def plan(self):
        # construct the state space we are planning in
        space = ob.RealVectorStateSpace(2)

        # set the bounds
        bounds = ob.RealVectorBounds(2)
        bounds.setLow(-4)
        bounds.setHigh(4)
        space.setBounds(bounds)
        # define a simple setup class
        ss = og.SimpleSetup(space)
        # set state validity checking for this space
        ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.isStateValid))

        # create a start state
        start = ob.State(space)
        start[0] = self.pl_req.start[0]
        start[1] = self.pl_req.start[1]

        # create a goal state
        goal = ob.State(space)
        goal[0] = self.pl_req.goal[0]
        goal[1] = self.pl_req.goal[1]

        # set the start and goal states;
        ss.setStartAndGoalStates(start, goal)

        # set sampler (optional; the default is uniform sampling)
        si = ss.getSpaceInformation()
        sampler = MyValidStateSampler(self.gmm_dist, self.pl_req)

        # use my sampler
        si.setValidStateSamplerAllocator(ob.ValidStateSamplerAllocator(sampler))

        # create a planner for the defined space
        planner = og.RRTConnect(si)
        ss.setPlanner(planner)

        # attempt to solve the problem within 20 seconds of planning time
        solved = ss.solve(20.0)
        solution_path = ss.getSolutionPath()

        ompl_solution = list(solution_path.getStates())
        solution = []

        for state in ompl_solution:
            np_state = np.zeros(2)
            np_state[0] = state[0]
            np_state[1] = state[1]
            solution.append(np_state)

        if solved:
            print("Found solution:")
            # print the path to screen
            print(ss.getSolutionPath())
            print(solution)
            """_, ax = plt.subplots(1, 1)
            # ax.scatter(samples[:, 0], samples[:, 1], s=1)
            # px = [node[0] for node in mean]
            # py = [node[1] for node in mean]
            # ax.scatter(px, py, color="red", s=10)
            # ax.scatter(req.start[0], req.start[1], color="green", s=60)
            # ax.scatter(req.goal[0], req.goal[1], color="blue", s=60)
            obstacles_space = pl.get_obstacle_space(req)
            obst_space_x = [ns[0] for ns in obstacles_space]
            obst_space_y = [ns[1] for ns in obstacles_space]
            ax.scatter(obst_space_x, obst_space_y, c="r", s=1)
            x = [ns[0] for ns in solution]
            y = [ns[1] for ns in solution]
            plt.plot(x, y)
            plt.show()"""

        else:
            print("No solution found")
        return self.get_count_max()

# max_count = MyValidStateSampler.get_count_max()
# print("maxcount:", max_count)

# if __name__ == '__main__':
# plan()
