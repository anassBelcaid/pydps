"""
Script for the first graph on the article which tests the recovery  of the top 
hat data using several window lenght 
"""

from pydps.datasets.datasets import load_top_hat
import torch
import matplotlib.pyplot as plt
from pydps.filter1d import DpsFilter1D


if __name__ == "__main__":
    
    #loading the data
    data, target = load_top_hat(128,0.5)

    #filtering 
    filt = DpsFilter1D(40, 0.8,8)

    line,grad,(left,right) = filt(data)
    
    plt.plot(target.numpy())
    plt.plot(data.numpy(),alpha=0.1,lw=0.5)
    plt.plot(grad.numpy(),label='grad')
    plt.plot(line.numpy(),label='line')
    plt.axhline(y=0.6,xmin=0,xmax=128)
    plt.legend()
    plt.show()
