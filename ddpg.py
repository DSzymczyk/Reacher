import numpy as np
import random

from ReplayBuffer import ReplayBuffer
from model import PolicyModel, ValueModel

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ddpg:

    def __init__(self, nS, nA, random_seed, load_checkpoint, checkpoint_prefix='', buffer_size=int(1e6), batch_size=512,
                 gamma=0.99, tau=1e-3, policy_lr=1e-4, value_lr=1e-3, weight_decay=0, learn_every=20, learns_number=10):
        self.state_size = nS
        self.action_size = nA
        self._checkpoint_prefix = checkpoint_prefix
        self.seed = random.seed(random_seed)

        self._online_policy_model = PolicyModel(self.state_size, self.action_size, random_seed).to(device)
        self._target_policy_model = PolicyModel(self.state_size, self.action_size, random_seed).to(device)
        self._online_value_model = ValueModel(self.state_size, self.action_size, random_seed).to(device)
        self._target_value_model = ValueModel(self.state_size, self.action_size, random_seed).to(device)

        if load_checkpoint:
            self.load_weights()

        self._policy_optimizer = optim.Adam(self._online_policy_model.parameters(), lr=policy_lr)
        self._value_optimizer = optim.Adam(self._online_value_model.parameters(), lr=value_lr, weight_decay=weight_decay)

        self._replay_buffer = ReplayBuffer(buffer_size, batch_size, random_seed)

        self.soft_update(self._online_value_model, self._target_value_model, 1.0)
        self.soft_update(self._online_policy_model, self._target_policy_model, 1.0)

        self._batch_size = batch_size
        self._gamma = gamma
        self._tau = tau
        self._learn_every = learn_every
        self._learns_number = learns_number
        self._step = 0

    def step(self, state, action, reward, next_state, done):
        self._replay_buffer.add(state, action, reward, next_state, done)
        self._step = (self._step + 1) % self._learn_every
        if (len(self._replay_buffer) > self._batch_size) and (self._step == 0):
            for _ in range(self._learns_number):
                experiences = self._replay_buffer.get_sample()
                self.learn(experiences, self._gamma)

    def act(self, state):
        state = torch.from_numpy(state).float().to(device)
        self._online_policy_model.eval()
        with torch.no_grad():
            action = self._online_policy_model(state).cpu().data.numpy()
        self._online_policy_model.train()
        return np.clip(action, -1, 1)

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        argmax_target_policy_model_action = self._target_policy_model(next_states)
        target_model_q_values = self._target_value_model(next_states, argmax_target_policy_model_action)

        target_q_values = rewards + (gamma * target_model_q_values * (1 - dones))
        expected_q_values = self._online_value_model(states, actions)
        value_loss = F.mse_loss(expected_q_values, target_q_values)

        self._value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._online_value_model.parameters(), 1)
        self._value_optimizer.step()

        argmax_online_policy_model_action = self._online_policy_model(states)
        policy_loss = -self._online_value_model(states, argmax_online_policy_model_action).mean()
        self._policy_optimizer.zero_grad()
        policy_loss.backward()
        self._policy_optimizer.step()

        self.soft_update(self._online_value_model, self._target_value_model, self._tau)
        self.soft_update(self._online_policy_model, self._target_policy_model, self._tau)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save_weights(self):
        torch.save(self._online_policy_model.state_dict(), f'weights/{self._checkpoint_prefix}policy_checkpoint.pth')
        torch.save(self._online_value_model.state_dict(), f'weights/{self._checkpoint_prefix}value_checkpoint.pth')

    def load_weights(self):
        self._online_policy_model.load_state_dict(torch.load(f'weights/{self._checkpoint_prefix}policy_checkpoint.pth'))
        self._online_policy_model.eval()
        self._target_policy_model.load_state_dict(torch.load(f'weights/{self._checkpoint_prefix}policy_checkpoint.pth'))
        self._target_policy_model.eval()
        self._online_value_model.load_state_dict(torch.load(f'weights/{self._checkpoint_prefix}value_checkpoint.pth'))
        self._online_value_model.eval()
        self._target_value_model.load_state_dict(torch.load(f'weights/{self._checkpoint_prefix}value_checkpoint.pth'))
        self._target_value_model.eval()
