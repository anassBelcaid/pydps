"""
simple functions to load specefic data
"""
from pydps.datasets.signals import random
import torch
from torch.distributions.normal import Normal
import numpy as np
import pydps.datasets
from os.path import abspath,dirname
from pydps.vis import plot_dataset
import matplotlib.pyplot as plt


#root folder to access file
root_fol=dirname(abspath(pydps.datasets.__file__))

def load_random_dataset(batch_size,signal_size,noise_scale=0.1):
    """
    Function to create a random piecewise constant signals 
    """

    targets = torch.zeros((batch_size,signal_size))

    #nois generator
    nois_gen = Normal(0,noise_scale)

    #data
    noise = nois_gen.sample((batch_size,signal_size))


    #number of discontinuities for each size
    num_dis  = np.random.choice(np.arange(2,10),batch_size,replace=True)

    for i in  range(batch_size):
        targets[i] = random(signal_size,num_dis=num_dis[i])
    
    data = targets + noise



    return data, targets


def load_molecular_motor(signal_size=1000):
    """load the molecular motor data presented in the article 
    An Energy Based Scheme for Reconstruction of Piecewise\
            Constant Signals observed in the Movement of Molecular Machine
    https://arxiv.org/abs/1504.07873
    :returns: a data size of single element 
    """

    #loading the array
    data = np.loadtxt(root_fol+"/noisy_molecular_motors.mm",skiprows=2)[:signal_size]
    target = np.loadtxt(root_fol+"/molecular_motors.mm",skiprows=2)[:signal_size]

    return torch.tensor(data[np.newaxis,:]), torch.tensor(target[np.newaxis,:])




def load_top_hat(signal_size=1000,sigma=0.2):
    """Load the top hat data presented in A.Blake and A.Zisserman book
    Visual Reconstruction 

    :signal_size:  signal size
    : sigma: noise standard variation
    :return:  data, target as tensors
    """

    #limits of the ones
    third,third2 = int(signal_size/3), int(2*signal_size/3)

    target = torch.zeros(signal_size)
    target[third:third2] = 1

    data =sigma*torch.randn(signal_size)+target

    return data, target
    



   


if __name__ == "__main__":

    data, target = load_top_hat(200)
    

