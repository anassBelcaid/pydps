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
import argparse
from sklearn.metrics import mean_squared_error

#default parser
parser = argparse.ArgumentParser()
parser.add_argument("-w", "--window", default=10, \
        help="window  of the filter",type=int)

parser.add_argument("-l", "--lamda", default=30, \
        help="Smoothness power",type=int)
parser.add_argument("-H", "--sensitivity", default=0.8, \
        help="sensibility of the filter",type=float)
parser.add_argument("-s", "--sigma", default=0.6, \
        help="noise standard deviation",type=float)
#parse arguments
args = parser.parse_args()



if __name__ == "__main__":
    
    #loading the data
    data, target = load_top_hat(128,args.sigma)

    #filtering 
    filt = DpsFilter1D(args.lamda, args.sensitivity,args.window)

    line,grad,(left,right) = filt(data)

    #rec geman
    rec_geman = reconstruct_geman_from_edge1d(data,line,args.lamda)
    #rec mean
    rec_mean = reconstruct_mean_from_edge1d(data,line)

    #rec median
    rec_median = reconstruct_median_from_edge1d(data,line)
    

    fig,axs = plt.subplots(1,3)

    #first plot
    axs[0].plot(target.numpy())
    axs[0].plot(data.numpy(),alpha=0.6,lw=0.5)
    axs[0].plot(rec_geman.numpy(),label='rec')
    err = mean_squared_error(target.numpy(),rec_geman.numpy())
    axs[0].set_title('geman err =%.2e'%err)
    axs[0].legend()


    #second plot
    axs[1].plot(target.numpy())
    axs[1].plot(data.numpy(),alpha=0.6,lw=0.5)
    axs[1].plot(rec_mean.numpy(),label='rec')
    err = mean_squared_error(target.numpy(),rec_mean.numpy())
    axs[1].set_title('mean err =%.2e'%err)


    #third plot
    axs[2].plot(target.numpy())
    axs[2].plot(data.numpy(),alpha=0.6,lw=0.5)
    axs[2].plot(rec_median.numpy(),label='rec')
    err = mean_squared_error(target.numpy(),rec_median.numpy())
    axs[2].set_title("median err =%.2e"%err)
    plt.show()
