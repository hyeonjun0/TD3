import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, Normal
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorNetwork(nn.Module):
    def init(self, state_dim, 
             action_dim, 
             hidden_dim, 
             max_action):
        super(ActorNetwork, self).__init__()
        
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)

        self.max_action = max_action

    def forward(self, state):
        x = torch.tanh(self.l1(state))
        x = torch.tanh(self.l2(x))
        x = torch.tanh(self.l3(x)) * self.max_action
        return x
    
class QCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(QCritic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)  
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        self.l4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3Agent(object):
    def __init__(
        self,
        env_with_dw,
        state_dim,
        action_dim,
        max_action,
        gamma=0.99,
        net_width=128,
        a_lr=1e-4,  # actor learning rate
        c_lr=1e-4,  # critic learning rate
        batch_size=256,
        policy_delay_freq=1
    ):
        self.actor = ActorNetwork(state_dim, action_dim, net_width, max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=a_lr)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic = QCritic(state_dim, action_dim, net_width).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=c_lr)
        self.critic_target = copy.deepcopy(self.critic)

        self.env_with_dw = env_with_dw  # dw: die or win
        self.action_dim = action_dim
        self.max_action = max_action
        self.gamma = gamma
        self.policy_noise = 0.2 * max_action
        self.noise_clip = 0.5 * max_action
        self.tau = 0.005  # delayed policy update
        self.batch_size = batch_size
        self.delay_counter = -1  # delayed policy update: reset
        self.delay_freq = policy_delay_freq

    def select_action(self, state):  # only used when interacting with the environment
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            action = self.actor(state)  # deterministic policy
        return action.cpu().numpy().flatten()  # torch (GPU) -> numpy (CPU)

    def train(self, replay_buffer):
        self.delay_counter += 1  # for delayed update of policy. -> delay_counter == delay_freq -> update.

        with torch.no_grad():
            states, actions, rewards, next_states, dw_masks = replay_buffer.sample(self.batch_size)
            # use clamp() to clip noise. 
			# torch.randn_like(a): returns a tensor with same size as input that filled with random numbers
			# from a standard normal distribution. 
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            smoothed_target_actions = (
                self.actor_target(next_states) + noise  # Noisy target action
            ).clamp(-self.max_action, self.max_action)  # Action regularization

        # Compute the target Q value by using the noisy target action.
        target_Q1, target_Q2 = self.critic_target(next_states, smoothed_target_actions)
        target_Q = torch.min(target_Q1, target_Q2)  # "Clipped" Double Q-learning: choose minimum estimate!

        if self.env_with_dw:
            target_Q = rewards + (1 - dw_masks) * self.gamma * target_Q  # dw: die or win
        else:
            target_Q = rewards + self.gamma * target_Q

        current_Q1, current_Q2 = self.critic(states, actions)

        q_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy update
        if self.delay_counter == self.delay_freq:
            a_loss = -self.critic.Q1(states, self.actor(states)).mean() #update actor
            self.actor_optimizer.zero_grad()
            a_loss.backward()
            self.actor_optimizer.step()

            # update the frozen target models 
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)  # slow update

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)  # slow update

            self.delay_counter = -1  # reset delay counter

    def save(self, env_name, episode):
        torch.save(self.actor.state_dict(), f"./model/{env_name}_actor{episode}.pth")
        torch.save(self.critic.state_dict(), f"./model/{env_name}_critic{episode}.pth")

    def load(self, env_name, episode):
        self.actor.load_state_dict(torch.load(f"./model/{env_name}_actor{episode}.pth"))
        self.critic.load_state_dict(torch.load(f"./model/{env_name}_critic{episode}.pth"))



