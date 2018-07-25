""" Training Everything Together 

vae, mdrnn, controller are overwritten each iteration
Will the data be overwritten each iteration? we cretainly think that the newer shaples should be more useful,
but it would be inefficient to throw them all away.
Ideally keep a curr dir, and a full_data dir, where we train on the curr dir for a bit with the recent data, and then use all data.

"""
import argparse
from os.path import join, exists
from os import mkdir

from new_vae import train_vae
from new_mdrnn import train_mdrnn
from new_controller import train_mdrnn


# for the final test
import torch
from utils.misc import RolloutGenerator



parser = argparse.ArgumentParser(description='VAE Trainer')
parser.add_argument('--iterations', type=int, default=10)
parser.add_argument('--vae_tot_epochs', type=int, default=200)
parser.add_argument('--mdrnn_tot_epochs', type=int, default=20)
parser.add_argument('--logdir', type=str, help='Directory where results are logged')


parser.add_argument('--PATH', type=str)
# logdir is top-level dir where everything for this trial is stored


# (where to save the data, or top level dir?)
# maybe there is one path, and inside this train, test, vae, mdrnn, anythting else of interest


# parser.add_argument('--logdir', type=str, help='Directory where results are logged')

# parser.add_argument('--noreload', action='store_true',
#                     help='Best model is not reloaded if specified')
# parser.add_argument('--nosamples', action='store_true',
#                     help='Does not save samples during training if specified')

# parser.add_argument('--traindir', type=str)
# parser.add_argument('--testdir', type=str, default=None)

args = parser.parse_args()
all_results = {}
all_results['best'] = []


gen_data(PATH=PATH, method=brown_noise)




for iteration in range(iterations):
	train_vae(logdir+'/train', logdir, epochs = int(vae_tot_epochs/iterations)) # maybe doing these things twice with two data dirs later
	train_mdrnn(logdir+'/train', logdir, epochs = int(mdrnn_tot_epochs/iterations))
	results = train_controller(logdir, n_samples=4, pop_size=4, target_return=400, display=True, max_workers=10)
	# train_explorer(PATH)

	all_results['best'].extend(results['best'])


device = torch.device('cpu')

generator = RolloutGenerator(args.logdir, device, 1000)

with torch.no_grad():
    generator.rollout(None)


