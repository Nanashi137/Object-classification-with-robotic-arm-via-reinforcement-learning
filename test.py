from agent import Resnet10, PPO, Actor, Critic
from ultilities import config_parser
import torch
from torch.utils.tensorboard import SummaryWriter
from agent import Resnet10

if __name__=="__main__":


    img_height = 224,
    img_width  = 224
    mode = "gray"
    n_joints   = 6

    backbone = Resnet10(height=img_height, width=img_height, mode=mode)

    actor  = Actor(arch=backbone, n_joints=n_joints)
    critic = Critic(arch=backbone)
    dummy_state = torch.randn(1, 1,224, 224)
    
    logging = SummaryWriter(log_dir="./temp") # initialize tensorboard summary writer

    ppo = PPO(actor=actor, critic=critic, logging=logging)


    means, stds = actor(dummy_state)
    print(means.shape)
    print(stds.shape)
    
    logging.close() # close the writer 



