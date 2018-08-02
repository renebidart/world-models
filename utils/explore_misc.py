""" 
Used for the exploration strategy

Exact same controller type is used for exploration, just with different weights learned. 
Difference is the reward for this controller is the sum of reconstruction error of vae over all time steps.

REWARD = -Reconstruction loss
"""
import numpy as np
import math
from os.path import join, exists

import torch
from torchvision import transforms
import torch.nn.functional as f

import gym
import gym.envs.box2d

from models import MDRNNCell, VAE, Controller
from models.mdrnn import gmm_loss
from torchvision import transforms
from models.vae import VAE
from models.mdrnn import MDRNN





# A bit dirty: manually change size of car racing env
gym.envs.box2d.car_racing.STATE_W, gym.envs.box2d.car_racing.STATE_H = 64, 64

# Hardcoded for now
ASIZE, LSIZE, RSIZE, RED_SIZE, SIZE =\
    3, 32, 256, 64, 64
BSIZE = 1 
SEQ_LEN = 32

# Same
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((RED_SIZE, RED_SIZE)),
    transforms.ToTensor()
])

def sample_continuous_policy(action_space, seq_len, dt):
    """ Sample a continuous policy.

    Atm, action_space is supposed to be a box environment. The policy is
    sampled as a brownian motion a_{t+1} = a_t + sqrt(dt) N(0, 1).

    :args action_space: gym action space
    :args seq_len: number of actions returned
    :args dt: temporal discretization

    :returns: sequence of seq_len actions
    """
    actions = [action_space.sample()]
    for _ in range(seq_len):
        daction_dt = np.random.randn(*actions[-1].shape)
        actions.append(
            np.clip(actions[-1] + math.sqrt(dt) * daction_dt,
                    action_space.low, action_space.high))
    return actions

def save_checkpoint(state, is_best, filename, best_filename):
    """ Save state in filename. Also save in best_filename if is_best. """
    torch.save(state, filename)
    if is_best:
        torch.save(state, best_filename)

def flatten_parameters(params):
    """ Flattening parameters.

    :args params: generator of parameters (as returned by module.parameters())

    :returns: flattened parameters (i.e. one tensor of dimension 1 with all
        parameters concatenated)
    """
    return torch.cat([p.detach().view(-1) for p in params], dim=0).cpu().numpy()

def unflatten_parameters(params, example, device):
    """ Unflatten parameters.

    :args params: parameters as a single 1D np array
    :args example: generator of parameters (as returned by module.parameters()),
        used to reshape params
    :args device: where to store unflattened parameters

    :returns: unflattened parameters
    """
    params = torch.Tensor(params).to(device)
    idx = 0
    unflattened = []
    for e_p in example:
        unflattened += [params[idx:idx + e_p.numel()].view(e_p.size())]
        idx += e_p.numel()
    return unflattened

def load_parameters(params, controller):
    """ Load flattened parameters into controller.

    :args params: parameters as a single 1D np array
    :args controller: module in which params is loaded
    """
    proto = next(controller.parameters())
    params = unflatten_parameters(
        params, controller.parameters(), proto.device)

    for p, p_0 in zip(controller.parameters(), params):
        p.data.copy_(p_0)



class RolloutGenerator(object):
    """ Utility to generate rollouts.

    Encapsulate everything that is needed to generate rollouts in the TRUE ENV
    using a controller with previously trained VAE and MDRNN.

    :attr vae: VAE model loaded from mdir/vae
    :attr mdrnn: MDRNN model loaded from mdir/mdrnn
    :attr controller: Controller, either loaded from mdir/ctrl or randomly
        initialized
    :attr env: instance of the CarRacing-v0 gym environment
    :attr device: device used to run VAE, MDRNN and Controller
    :attr time_limit: rollouts have a maximum of time_limit timesteps
    """
    def __init__(self, mdir, device, time_limit, explorer=False):
        """ Build vae, rnn, controller and environment. """
        self.explorer = explorer

        # Load controllers
        vae_file, rnn_file, ctrl_file = \
            [join(mdir, m, 'best.tar') for m in ['vae', 'mdrnn', 'ctrl']]

        if self.explorer:
            ctrl_file = join(mdir, 'exp', 'best.tar')

        assert exists(vae_file) and exists(rnn_file),\
            "Either vae or mdrnn is untrained."

        vae_state, rnn_state = [
            torch.load(fname, map_location={'cuda:0': str(device)})
            for fname in (vae_file, rnn_file)]

        for m, s in (('VAE', vae_state), ('MDRNN', rnn_state)):
            print("Loading {} at epoch {} "
                  "with test loss {}".format(
                      m, s['epoch'], s['precision']))

        self.vae = VAE(3, LSIZE).to(device)
        self.vae.load_state_dict(vae_state['state_dict'])


        # MDRNNCell
        self.mdrnn = MDRNNCell(LSIZE, ASIZE, RSIZE, 5).to(device)
        self.mdrnn.load_state_dict(
            {k.strip('_l0'): v for k, v in rnn_state['state_dict'].items()})

        self.controller = Controller(LSIZE, RSIZE, ASIZE).to(device)

        # load controller if it was previously saved
        if exists(ctrl_file):
            ctrl_state = torch.load(ctrl_file, map_location={'cuda:0': str(device)})
            print("Loading Controller with reward {}".format(
                ctrl_state['reward']))
            self.controller.load_state_dict(ctrl_state['state_dict'])

        self.env = gym.make('CarRacing-v0')
        self.device = device

        self.time_limit = time_limit


        self.mdrnn_notcell = MDRNN(LSIZE, ASIZE, RSIZE, 5)
        self.mdrnn_notcell.to(device)
        self.mdrnn_notcell.load_state_dict(rnn_state['state_dict'])

#####$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    # VERY LAZY. Copied from the other trainmdrnn file
    # from trainmdrnn import get_loss, to_latent
    def to_latent(self, obs, next_obs):
        """ Transform observations to latent space.

        :args obs: 5D torch tensor (BSIZE, SEQ_LEN, ASIZE, SIZE, SIZE)
        :args next_obs: 5D torch tensor (BSIZE, SEQ_LEN, ASIZE, SIZE, SIZE)

        :returns: (latent_obs, latent_next_obs)
            - latent_obs: 4D torch tensor (BSIZE, SEQ_LEN, LSIZE)
            - next_latent_obs: 4D torch tensor (BSIZE, SEQ_LEN, LSIZE)
        """

        with torch.no_grad():
            obs, next_obs = [
                f.upsample(x.view(-1, 3, SIZE, SIZE), size=RED_SIZE,
                           mode='bilinear', align_corners=True)
                for x in (obs, next_obs)]

            (obs_mu, obs_logsigma), (next_obs_mu, next_obs_logsigma) = [
                self.vae(x)[1:] for x in (obs, next_obs)]

            SEQ_LEN=1

            latent_obs, latent_next_obs = [
                (x_mu + x_logsigma.exp() * torch.randn_like(x_mu)).view(BSIZE, SEQ_LEN, LSIZE)
                for x_mu, x_logsigma in
                [(obs_mu, obs_logsigma), (next_obs_mu, next_obs_logsigma)]]

        return latent_obs, latent_next_obs

    def mdrnn_exp_reward(self, latent_obs, action, reward, latent_next_obs, hidden):
        """  # REMOVE TERMINAL

        Compute losses.

        The loss that is computed is:
        (GMMLoss(latent_next_obs, GMMPredicted) + MSE(reward, predicted_reward) +
             BCE(terminal, logit_terminal)) / (LSIZE + 2)
        The LSIZE + 2 factor is here to counteract the fact that the GMMLoss scales
        approximately linearily with LSIZE. All losses are averaged both on the
        batch and the sequence dimensions (the two first dimensions).

        :args latent_obs: (BSIZE, SEQ_LEN, LSIZE) torch tensor
        :args action: (BSIZE, SEQ_LEN, ASIZE) torch tensor
        :args reward: (BSIZE, SEQ_LEN) torch tensor
        :args latent_next_obs: (BSIZE, SEQ_LEN, LSIZE) torch tensor

        :returns: dictionary of losses, containing the gmm, the mse, the bce and
            the averaged loss.
        """

        mus, sigmas, logpi, rs, ds, next_hidden = self.mdrnn(action, latent_obs, hidden)
        gmm = gmm_loss(latent_next_obs, mus, sigmas, logpi)
        # bce = f.binary_cross_entropy_with_logits(ds, terminal)
        mse = f.mse_loss(rs, reward)
        loss = (gmm + mse) / (LSIZE + 2)
        return loss.squeeze().cpu().numpy()

    # def recon_error_reward(self, obs, hidden, obs_new):
    #     print('recon_error_reward')
    #     """Find out how good the reconstruction was.
    #     Encoding the vae to get mu and the controller action is deterministic, so its fine to be duplicated
    #     ??? maybe remove this and the above function because of unnecessary duplication
    #     """
    #     # obs_new = torch.from_numpy(np.moveaxis(obs_new, 2, 0).copy()).unsqueeze(0).to(self.device).type(torch.cuda.FloatTensor)
    #     # obs = obs.to(self.device).type(torch.cuda.FloatTensor)

    #     _, latent_mu, _ = self.vae(obs)
    #     action = self.controller(latent_mu, hidden[0])

    #     mus, sigmas, logpi, r, d, next_hidden = self.mdrnn(action, latent_mu, hidden)
    #     print('mus.size()', mus.size())
    #     print('sigmas.size()', sigmas.size())
    #     print('logpi.size()', logpi.size())
    #     print('r.size()', r.size())
    #     print('d.size()', d.size())
    #     print('next_hidden.size() [0], [1]', next_hidden[0].size(), next_hidden[1].size())


    #     recon_x = self.vae.decoder(mus.squeeze()).type(torch.cuda.FloatTensor) # ??? this is just mu, right? Still a bit confused
    #     print('obs_new.size()', obs_new.size())
    #     print('recon_x.size()', recon_x.size())

    #     # reward = -1*((recon_x - obs_new) ** 2).mean()
    #     reward = -1*F.mse_loss(recon_x, obs_new).item()


    def rollout(self, params, render=False):
        """ Execute a rollout and return reward

        Load :params: into the controller and execute a single rollout. This
        is the main API of this class.

        :args params: parameters as a single 1D np array

        :returns: minus cumulative reward if ctrl mode, cumulative recon_error if exp mode
        """
        # copy params into the controller
        if params is not None:
            load_parameters(params, self.controller)

        obs = self.env.reset()

        # This first render is required !
        self.env.render()

        hidden = [
            torch.zeros(1, RSIZE).to(self.device)
            for _ in range(2)]

        cumulative = 0
        i = 0
        while True:
            obs = transform(obs).unsqueeze(0).to(self.device)

            # GET ACTION
            _, latent_mu, _ = self.vae(obs)
            action = self.controller(latent_mu, hidden[0])
            _, _, _, _, _, next_hidden = self.mdrnn(action, latent_mu, hidden)
            action = action.squeeze().cpu().numpy()

            next_obs, reward, done, _ = self.env.step(action)

            if self.explorer:
                latent_obs, latent_next_obs = self.to_latent(obs.unsqueeze(0), transform(next_obs).unsqueeze(0).to(self.device))
                action = torch.from_numpy(action).unsqueeze(0)
                latent_obs = latent_obs.to(self.device).squeeze().unsqueeze(0)
                latent_next_obs = latent_next_obs.to(self.device).squeeze().unsqueeze(0)
                action = action.to(self.device)
                reward = torch.from_numpy(np.array(reward)).unsqueeze(0).type(torch.cuda.FloatTensor)
                reward = self.mdrnn_exp_reward(latent_obs, action, reward, latent_next_obs, hidden)

            obs = next_obs
            hidden = next_hidden

            if render:
                self.env.render()

            cumulative += reward
            if done or i > self.time_limit:
                return - cumulative
            i += 1



class RolloutGeneratorSingle(object):
    def __init__(self, mdir, device, controller_model):
        """ Run one step. 
        Load VAE and MDRNN from files
        Take the controller (exp/ctrl) an an input, so we can easily change stuff inside the other file.
         """
        self.controller = controller_model.to(device)

        # Load controllers
        vae_file, rnn_file, ctrl_file = \
            [join(mdir, m, 'best.tar') for m in ['vae', 'mdrnn', 'ctrl']]

        assert exists(vae_file) and exists(rnn_file),\
            "Either vae or mdrnn is untrained."

        vae_state, rnn_state = [
            torch.load(fname, map_location={'cuda:0': str(device)})
            for fname in (vae_file, rnn_file)]

        for m, s in (('VAE', vae_state), ('MDRNN', rnn_state)):
            print("Loading {} at epoch {} "
                  "with test loss {}".format(
                      m, s['epoch'], s['precision']))

        self.vae = VAE(3, LSIZE).to(device)
        self.vae.load_state_dict(vae_state['state_dict'])


        # MDRNNCell
        self.mdrnn = MDRNNCell(LSIZE, ASIZE, RSIZE, 5).to(device)
        self.mdrnn.load_state_dict(
            {k.strip('_l0'): v for k, v in rnn_state['state_dict'].items()})


    def single_step(obs, hidden):
        if params is not None:
            load_parameters(params, self.controller)

        obs = transform(obs).unsqueeze(0).to(self.device)

        # GET ACTION
        _, latent_mu, _ = self.vae(obs)
        action = self.controller(latent_mu, hidden[0])
        _, _, _, _, _, next_hidden = self.mdrnn(action, latent_mu, hidden)
        action = action.squeeze().cpu().numpy()

        return action, next_obs, next_hidden


