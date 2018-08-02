""" Training Everything Together 

vae, mdrnn, controller are overwritten each iteration
Will the data be overwritten each iteration? we cretainly think that the newer shaples should be more useful,
but it would be inefficient to throw them all away.
Ideally keep a curr dir, and a full_data dir, where we train on the curr dir for a bit with the recent data, and then use all data.

The controller and explorer load the previous best one every iteration.
"""
import argparse
from os.path import join, exists
from os import mkdir

from data.new_data_gen import run_generate_data
from new_vae import train_vae
from new_mdrnn import train_mdrnn
from subprocess import call

import torch
from utils.misc import RolloutGenerator

parser = argparse.ArgumentParser()
parser.add_argument('--logdir', type=str, help='top-level dir where everything for this trial is stored')
parser.add_argument('--iterations', type=int, default=10)
parser.add_argument('--vae_tot_epochs', type=int, default=20) # Original world models used 10
parser.add_argument('--mdrnn_tot_epochs', type=int, default=20) # Original world models used ???
parser.add_argument('--total_rollouts', type=int, default=10000) # same default as Original world models
parser.add_argument('--target_return', type=int, default=600)
parser.add_argument('--ctrl_tot_epochs', type=int, default=100)
parser.add_argument('--exp_tot_epochs', type=int, default=100)
parser.add_argument('--use_ctrl_exp', action='store_true')
parser.add_argument('--noise_type', type=str, default='white')
parser.add_argument('--exp_prob', type=float, default=.5)
parser.add_argument('--randomness_factor', type=float, default=.5)
args = parser.parse_args()

# make data directory:
logdir = args.logdir
data_dir = join(logdir, 'train')
if not exists(data_dir):
    mkdir(data_dir)
print('data_dir', data_dir)


# The controller must be run with xvfb-run, so do this the bad way calling command line from python.
def train_controller(iteration, epochs):
    cmd = ['xvfb-run', '-s', '"-screen 0 1400x900x24"']
    cmd += ["python", "train_ctrl_exp.py"]
    cmd += ["--logdir", logdir]
    cmd += ["--n-samples", str(4)]
    cmd += ["--pop-size", str(4)]
    # cmd += ["--target-return", str(args.target_return)]
    # cmd += ["--display"]
    cmd += ["--max-workers", str(10)]
    cmd += ["--iteration", str(iteration)]
    cmd += ["--max-epochs", str(epochs)]

    cmd = " ".join(cmd)
    print('train_controller cmd: ', cmd)
    call(cmd, shell=True)

def train_explorer(iteration, epochs):
    cmd = ['xvfb-run', '-s', '"-screen 0 1400x900x24"']
    cmd += ["python", "train_ctrl_exp.py"]
    cmd += ["--logdir", logdir]
    cmd += ["--n-samples", str(4)]
    cmd += ["--pop-size", str(4)]
    # cmd += ["--target-return", str(args.target_return)]
    # cmd += ["--display"]
    cmd += ["--max-workers", str(10)]
    cmd += ["--iteration", str(iteration)]
    cmd += ["--explore"]
    cmd += ["--max-epochs", str(epochs)]
    cmd = " ".join(cmd)
    print('train_explorer cmd: ', cmd)
    call(cmd, shell=True)

# generate some data before anything is trained, also makes a vae, etc so that the run_generate_data will work
run_generate_data(rollouts=int(args.total_rollouts/args.iterations), logdir=logdir, threads=10, noise_type='brown', 
	exp_prob=1, randomness_factor=1, use_ctrl_exp=False, device="cuda:0")

for iteration in range(args.iterations):
	print('Iteration', iteration)
	run_generate_data(rollouts=int(args.total_rollouts/args.iterations), logdir=logdir, threads=10, noise_type=args.noise_type,
                   exp_prob=args.exp_prob, randomness_factor=args.randomness_factor, use_ctrl_exp=True, device="cuda:0")

	train_vae(logdir, traindir=data_dir, epochs = int(args.vae_tot_epochs/args.iterations)) # maybe doing these things twice with two data dirs later
	train_mdrnn(logdir, data_dir, epochs = int(args.mdrnn_tot_epochs/args.iterations))
	train_controller(iteration, int(args.ctrl_tot_epochs/args.iterations))
	train_explorer(iteration, int(args.exp_tot_epochs/args.iterations))

# test performance, again calling from command line because can't do xvfb-run in python
cmd = ['xvfb-run', '-s', '"-screen 0 1400x900x24"']
cmd += ["python", "test_controller.py"]
cmd += ["--logdir", logdir]
cmd = " ".join(cmd)
print('test_controller cmd: ', cmd)
call(cmd, shell=True)



