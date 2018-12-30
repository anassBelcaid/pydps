"""
Module to vizualize the results and dataset of the alrorithm
"""
import matplotlib.pyplot as plt 
import numpy as np
import torch

from sklearn.metrics import accuracy_score

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


def plot_edge_detection1d(noised,edge,gt=None,gradient=None):
    """Plot the result of the edge detection 

    :noised: noised signal (torch tensor)
    :edge: edge result (torch tensor of 1 and 0)
    :gt:  ground truth  ( torch tensor )
    :gradient: Gradient results ( troch tensor
    """

    fig,ax= plt.subplots(1,3,figsize=(12,6))

    #first plot 
    ax[0].plot(noised.numpy(),lw=0.4,alpha=0.4)
    if(gt is not None):
        ax[0].plot(gt.numpy(),lw=2)

    #second plot
    ax[1].stem(edge)
    if(gradient is not None):
        ax[2].plot(gradient.numpy())
        ax[2].set_title('gradient')

    if(gt is not None):
        gt_line  = gt[1:]- gt[0:-1]
        gt_line[gt_line !=0]= 1
        ax[1].stem(gt_line,label='ground truth')
        accuracy = accuracy_score(gt_line.numpy(),edge.numpy())
        ax[1].set_title('accuracy= %.2f'%accuracy)

    plt.show()
