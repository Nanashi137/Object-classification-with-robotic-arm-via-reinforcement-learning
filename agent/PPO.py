import torch 
import torch.nn as nn 

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
        self.exp = torch.exp


    def _weight_init(self, ):

        pass 

    def forward(self, state: torch.Tensor):
        features = torch.flatten(self.encoder(state), start_dim=1)

        action_means = self.mean_head(features)
        action_stds  = self.exp(self.std_head(features))
 
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
    
    def _count_params(self): 
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class PPO():
    def __init__(self, actor: nn.Module, critic: nn.Module, lr: float=3e-4) -> None:
        
        # initializing ppo parameters 
        self.actor = actor
        self.critic = critic 
        self.discount_factor 
        self.Lambda
        self.clip

        # loss coefficient 
        self.values_loss_coeff
        self.entropy_loss_coeff
        
        # initializing optimizer 
        self.lr = lr
        self.optimizer = torch.optim.Adam(params= list(self.actor.parameters()) + list(self.critic.parameters()), lr=self.lr)

    def _update(self, states: torch.Tensor, actions: torch.Tensor, log_probs_old, advantages, returns): 
        
        for i in range (self.epoch): 
            means, stds = self.actor(states)

            dists = torch.distributions.Normal(loc=means, scale=stds)
            log_probs_new = dists.sample(actions).sum()
            entropy = dists.entropy()
            
            values = self.critic(states)
            
            ratio = torch.exp(log_probs_new - log_probs_old)
            surr1 = ratio*advantages
            surr2 = torch.clamp(ratio, 1-self.clip, 1+self.clip)

            # Defining losses 
            policy_loss  = -torch.min(surr1, surr2).mean()
            value_loss   = nn.MSELoss()(values, returns)
            entropy_loss = -entropy.mean()

            loss = policy_loss + self.values_loss_coeff*value_loss + self.entropy_loss_coeff*entropy_loss


            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


    def _sample_action(self, state: torch.Tensor):
        
        means, stds = self.actor(state)

        dists = torch.distributions.Normal(loc=means, scale=stds)

        action = dists.sample()

        return action 



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

