
# coding: utf-8

import torch 
from pydps import filter1d
from pydps.datasets import datasets
from pydps.vis import plot_edge_detection1d
import numpy as np


if __name__ == "__main__":
    #simple data
    data, target = datasets.load_random_dataset(20,200,0.4)

    #visualizing the data
    datasets.plot_dataset(data,targets=target,indices=np.random.choice(np.arange(20),12))
    

    #preparing the filter
    window_size = 10
    lam = 30
    threshold = 0.2

    #Creating the filter
    filt = filter1d.DpsFilter1D(lam,threshold,window_size)


    #filtering the first signal
    line, grad, (left,right) = filt(data[1])
    plot_edge_detection1d(data[1],line,gt=target[1],gradient=grad)



