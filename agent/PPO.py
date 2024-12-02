import torch 
import torch.nn as nn 
from torch.utils.tensorboard import SummaryWriter

import os

from .ResNet import Resnet10


class MLP(nn.Module):
    def __init__(self, *args, **kwargs):
        super(MLP, self).__init__()

class Actor(nn.Module):
    
    def __init__(self, arch: nn.Module, n_joints: int) -> None:
        super(Actor, self).__init__()
 
        self.encoder = arch 
        self.height = self.encoder.height 
        self.width = self.encoder.width
        self.n_joints = n_joints

        self.hidden_dim = self.encoder.hidden_dim_mlp

        self.mean_head = nn.Linear(in_features=self.hidden_dim, out_features=self.n_joints)
        self.std_head  = nn.Linear(in_features=self.hidden_dim, out_features=self.n_joints)
        self.softplus = nn.Softplus


    def forward(self, state: torch.Tensor):
        features = torch.flatten(self.encoder(state), start_dim=1)

        action_means = self.mean_head(features)
        action_stds  = self.softplus(self.std_head(features))
 
        return action_means, action_stds

    def _count_params(self): 
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

        
class Critic(nn.Module):

    def __init__(self, arch: nn.Module):
        super(Critic, self).__init__()

        self.encoder = arch 
        self.height = self.encoder.height 
        self.width = self.encoder.width

        self.hidden_dim = self.encoder.hidden_dim_mlp

        self.head = nn.Linear(in_features=self.hidden_dim, out_features=1)

    def forward(self, state: torch.Tensor):
        features = torch.flatten(self.encoder(state), start_dim=1)

        value  = self.head(features)

        return value
    
    def count_params(self): 
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class PPO():
    def __init__(self, actor: nn.Module, critic: nn.Module, lr: float=3e-4, values_loss_coeff: float=1.0, entropy_loss_coeff: float=0.01, log_dir: str="./agent_log") -> None:
        
        # initializing ppo parameters 
        self.actor = actor
        self.critic = critic 
        self.clip
        

        # loss coefficient 
        self.values_loss_coeff = values_loss_coeff
        self.entropy_loss_coeff = entropy_loss_coeff
        
        # initializing optimizer 
        self.lr = lr
        self.optimizer = torch.optim.Adam(params= list(self.actor.parameters()) + list(self.critic.parameters()), lr=self.lr)

        # logging 
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)  # TensorBoard writer

        self.step = 0  # global step counter for logging

    def update(self, states: torch.Tensor, actions: torch.Tensor, log_probs_old, advantages, returns): 
        
        for i in range (self.epoch): 
            means, stds = self.actor(states)

            dists = torch.distributions.Normal(loc=means, scale=stds)
            log_probs_new = dists.sample(actions).sum(dim=1).mean(dim=0)
            entropy = dists.entropy().mean()
            
            values = self.critic(states)
            
            ratio = torch.exp(log_probs_new - log_probs_old)
            surr1 = ratio*advantages
            surr2 = torch.clamp(ratio, 1-self.clip, 1+self.clip)

            # defining losses 
            policy_loss  = -torch.min(surr1, surr2).mean()
            value_loss   = nn.MSELoss()(values, returns)
            entropy_loss = -entropy

            loss = policy_loss + self.values_loss_coeff*value_loss + self.entropy_loss_coeff*entropy_loss

            # backprop 
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # logging
            self.writer.add_scalar('Loss/Policy', policy_loss.item(), self.step)
            self.writer.add_scalar('Loss/Value', value_loss.item(), self.step)
            self.writer.add_scalar('Loss/Entropy', entropy_loss.item(), self.step)
            self.writer.add_scalar('Values/Mean', values.mean().item(), self.step)
            self.writer.add_scalar('Values/Std', values.std().item(), self.step)
            
            self.step += 1

    def sample_action(self, state: torch.Tensor):
        """Sample action given state"""
        means, stds = self.actor(state)

        dists = torch.distributions.Normal(loc=means, scale=stds)

        action = dists.sample()

        return action 

    def save(self, path):
        """Save the model parameters."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)

    def load(self, path):
        """Load the model parameters."""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


if __name__=="__main__":

    actor  = Actor(arch=Resnet10(height=224, width=224, mode="gray"), n_joints=6)
    critic = Critic(arch=Resnet10(height=224, width=224, mode="gray"))

    # print(actor._count_params())
    # print(critic._count_params())

    actor.eval(), critic.eval()
    dummy_state = torch.randn(5, 1, 224, 224)

    action_means, action_stds = actor(dummy_state)
    value = critic(dummy_state)

    print(action_means)
    print(action_stds)
    print(value)

