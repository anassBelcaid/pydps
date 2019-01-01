"""
Script for the first graph on the article which tests the recovery  of the top 
hat data using several window lenght 
"""

from pydps.datasets.datasets import load_top_hat
import torch
import matplotlib.pyplot as plt
from pydps.filter1d import DpsFilter1D
from pydps.reconstruction import reconstruct_mean_from_edge1d
from pydps.reconstruction import reconstruct_median_from_edge1d
from pydps.reconstruction import reconstruct_geman_from_edge1d


if __name__ == "__main__":
    
    #loading the data
    data, target = load_top_hat(128,0.5)

    #filtering 
    filt = DpsFilter1D(20, 0.8,8)

    line,grad,(left,right) = filt(data)

    recoverd = reconstruct_geman_from_edge1d(data,line,40)
    
    plt.plot(target.numpy())
    plt.plot(data.numpy(),alpha=0.6,lw=0.5)
    plt.plot(recoverd.numpy(),label='rec')
    plt.legend()
    plt.show()
