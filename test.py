from agent import Resnet10, PPO, Actor, Critic
from ultilities import config_parser
import torch
from torch.utils.tensorboard import SummaryWriter


if __name__=="__main__":


    img_height = 224,
    img_width  = 224
    mode = "gray"
    n_joints   = 6

    backbone = Resnet10(height=img_height, width=img_height, mode=mode)

    actor  = Actor(arch=backbone, n_joints=n_joints)
    critic = Critic(arch=backbone)
    
    logging = SummaryWriter(log_dir="./temp")

    ppo = PPO(actor=actor, critic=critic, logging=logging)

    dummy_state = torch.randn(5, 1,224, 224)

    means, stds = actor(dummy_state)
    print(means.shape)
    print(stds.shape)


