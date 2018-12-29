"""
simple functions to load specefic data
"""
from pydps.datasets.signals import random
import torch
from torch.distributions.normal import Normal
import numpy as np

from pydps.vis import plot_dataset

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




if __name__ == "__main__":
    
    X,y = load_random_dataset(10,200)
    plot_dataset(X,y,indices = [0,1,4,6,3,8])

