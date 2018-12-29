"""
simple functions to load specefic data
"""
from pydps.datasets.signals import random
import torch
from torch.distributions.normal import Normal
import numpy as np
import matplotlib.pyplot as plt


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



def plot_dataset(data,targets, indices=None):
    """
    plot a set of examples indexec by indices from the  dataset
    """

    #prior processing of indices

    if(indices is None):
        indices = np.arange(len(data))
    print(indices)

    #number  of lines for the subplot
    N = len(indices)
    num_lin =  int(np.sqrt(N))
    num_col = N // num_lin+1

    fig, axs = plt.subplots(num_lin,num_col,sharex =True,figsize=(12,8))


    if(num_col==1): 
        for index in range(N):
            k = indices[index]
            axs[index//num_col].plot(data[k].numpy())
            axs[index//num_col].plot(targets[k].numpy())
    

    else:
        for index in range(N):
            k = indices[index]
            ax = axs[index//num_col, index%num_col]
            ax.plot(data[k].numpy())
            ax.plot(targets[k].numpy())
            # ax.set_title('Example %d'%k)

    plt.show()

if __name__ == "__main__":
    
    X,y = load_random_dataset(10,200)
    plot_dataset(X,y,indices = [0,1,4,6,3,8])

