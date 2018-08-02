"""

"""
import argparse
import random
from os.path import join, exists
import gym
import numpy as np

import torch
import torch.nn.functional as f

from models import Controller
from utils.explore_misc import ASIZE, RSIZE, LSIZE
from utils.explore_misc import load_parameters
from utils.explore_misc import flatten_parameters
from utils.explore_misc import RolloutGeneratorSingle
from utils.explore_misc import sample_continuous_policy


def ctrl_exp_gen_data(rollouts, datadir, logdir, noise_type, device, use_ctrl_exp, exp_prob=.5, randomness_factor=.1):
    """ 
    randomness factor is the multiple we will multiply the current standard deviation by to get the std for the normal disnt std.
    This help because it is more resonable. Really should be updating based on parameter distances over updates, but whatever.
    ** read the openai parameter thing

    Uses fixed parameters for vae and mdrnn, but maybe change

    All the if use_ctrl_exp: should be switched to having the random thing inside a module, or at least consistent with the explorer way.
    """
    assert exists(logdir), "The directory does not exist..."
    exp_prob = float(exp_prob)

    env = gym.make("CarRacing-v0")
    seq_len = 1000

    if use_ctrl_exp:
        a_rollout = []

        #### Load controller and explorer
        ctrl_file = join(logdir, 'ctrl', 'best.tar')
        exp_file = join(logdir, 'exp', 'best.tar')

        controller = Controller(LSIZE, RSIZE, ASIZE).to(device)
        explorer = Controller(LSIZE, RSIZE, ASIZE).to(device)

        if exists(ctrl_file):
            ctrl_state = torch.load(ctrl_file, map_location={'cuda:0': str(device)})
            print("Loading Controller with reward {}".format(
                ctrl_state['reward']))
            controller.load_state_dict(ctrl_state['state_dict'])

        if exists(exp_file):
            exp_state = torch.load(exp_file, map_location={'cuda:0': str(device)})
            print("Loading Explorer with reward {}".format(
                exp_state['reward']))
            explorer.load_state_dict(exp_state['state_dict'])

        # Make the generators (this is unnecessary, shoul dbe organized some other way)
        ctrl_gen = RolloutGeneratorSingle(logdir, device, controller)
        exp_gen = RolloutGeneratorSingle(logdir, device, explorer)

        # for parameter noise exploration
        def update_params_noise(model, randomness_factor):
            def gaussian(ins, stddev=std):
                return ins + Variable(torch.randn(ins.size()).cuda() * stddev)

            all_params = []
            controller_new = controller
            for name, param in controller.named_parameters():
                all_params.append(param)

            std = np.std(np.array(params))
            print('Parameter mean: ' , np.mean(np.array(params)))
            print('Parameter std: ' , std)

            std = std*randomness_factor
            controller_new.apply(gaussian)
            return controller_new


    for i in range(rollouts):
        env.reset()
        env.env.viewer.window.dispatch_events()

        s_rollout = []
        r_rollout = []
        d_rollout = []

        if use_ctrl_exp:
            # randomize the explorer and controller
            explorer_new = update_params_noise(explorer, randomness_factor)
            controller_new = update_params_noise(controller, randomness_factor)

            # initialize the hidden state for the model:
            hidden = [
                torch.zeros(1, RSIZE).to(device)
                for _ in range(2)]

        else:
            if noise_type == 'white':
                a_rollout = [env.action_space.sample() for _ in range(seq_len)]
            elif noise_type == 'brown':
                a_rollout = sample_continuous_policy(env.action_space, seq_len, 1. / 50)


        t = 0
        while True:

            if use_ctrl_exp:
                # explore or exploit:
                if random.uniform(0, 1)<exp_prob:
                    action, obs, hidden = ctrl_gen(obs, hidden)
                else:
                    action, obs, hidden  = exp_gen(obs, hidden)
                a_rollout.append(action)
            else:
                action = a_rollout[t]

            t += 1
            s, r, done, _ = env.step(action)
            env.env.viewer.window.dispatch_events()
            s_rollout += [s]
            r_rollout += [r]
            d_rollout += [done]
            if done:
                print("> End of rollout {}, {} frames...".format(i, len(s_rollout)))
                np.savez(join(datadir, 'rollout_{}'.format(i)),
                         observations=np.array(s_rollout),
                         rewards=np.array(r_rollout),
                         actions=np.array(a_rollout),
                         terminals=np.array(d_rollout))
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rollouts', type=int, help="Number of rollouts")
    parser.add_argument('--dir', type=str, help="Where to place rollouts")
    parser.add_argument('--logdir', type=str, help="Top level dir where everything is stored")
    parser.add_argument('--noise_type', type=str, choices=['white', 'brown'], default='brown')
    parser.add_argument('--exp_prob', type=float, default=.5)
    parser.add_argument('--randomness_factor', type=float, default=.1)
    parser.add_argument('--device', type=str, default="cu")
    parser.add_argument('--use_ctrl_exp', action='store_true')

    args = parser.parse_args()
    ctrl_exp_gen_data(rollouts=args.rollouts, datadir=args.dir, logdir=args.logdir, noise_type=args.noise_type, device=args.device, 
                      use_ctrl_exp=args.use_ctrl_exp, exp_prob=args.exp_prob, randomness_factor=args.randomness_factor)
