""" Test controller 
Needs to be run using xvfb-run -s "-screen 0 1400x900x24"
"""
import argparse
from os.path import join, exists
from utils.misc import RolloutGenerator
import torch
import numpy as np

### ADD IN SOMETHING TO DO AVG AND STD OF A FEW TRIALS
parser = argparse.ArgumentParser()
parser.add_argument('--logdir', type=str, help='Where models are stored.')
args = parser.parse_args()

ctrl_file = join(args.logdir, 'ctrl', 'best.tar')

assert exists(ctrl_file),\
    "Controller was not trained..."

device = torch.device('cpu')

rewards = []

for i in range(10):
	generator = RolloutGenerator(args.logdir, device, 1000)
	with torch.no_grad():
	    final_reward = generator.rollout(None)
	rewards.append(final_reward)


print('last_reward', final_reward)
print('mean reward', np.mean(np.array(rewards)))
print('std reward', np.std(np.array(rewards)))



np.save(args.logdir +'/final_reward.npy', final_reward)
