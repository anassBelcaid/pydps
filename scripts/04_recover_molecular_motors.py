"""
Script to generate to recover the molecular motor data and report 
the MSE loss
"""

from pydps.datasets.datasets import load_molecular_motor
from pydps.filter1d import DpsFilter1D
from pydps.reconstruction import reconstruct_mean_from_edge1d
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix
from pydps.metrics import ball_f1score
from pydps.helper_functions import save_csv_table

#default argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--size", default=40000,\
        help="size of thesignal",type=int)
parser.add_argument("-w", "--window", default=30, \
        help="window  of the filter",type=int)

parser.add_argument("-l", "--lamda", default=100, \
        help="Smoothness power",type=int)
parser.add_argument("-H", "--sensitivity", default=0.7, \
        help="sensibility of the filter",type=float)

parser.add_argument("-e", "--delta", default=10,\
        help="delta x forcomputing the f1  score",type=int)

parser.add_argument("-j", "--jump", default=10,type=int,\
        help="Jumps to save the data to avoid huge data")
args = parser.parse_args()

if __name__ == "__main__":
    #loading the data
    size = args.size
    data,target = load_molecular_motor(size)
    data.squeeze_(), target.squeeze_()


    #applyting the filter
    fil = DpsFilter1D(args.lamda, args.sensitivity, args.window)

    edge, grad, (left,right) = fil(data) 

    #reconstruct by mean
    rec =reconstruct_mean_from_edge1d(data,edge)

    
    #error
    error =mean_squared_error(target.numpy(), rec.numpy())

    #printing the classification report
    gt_edge = target[1:]!=target[:-1]
    print(confusion_matrix(gt_edge.numpy(),edge.numpy(),labels=[0,1]))
    f1score= ball_f1score(gt_edge,edge,args.delta)

    #plotting
    J  = args.jump
    plt.plot(data.numpy()[::J],alpha=0.4,lw=0.3,label='nois')
    plt.plot(target.numpy()[::J],label='pwc_data')
    plt.plot(rec.numpy()[::J],'.',ms='0.2',label='dps')
    plt.title('Reconstruction with Dps Filter, err=%.2e,\
            f1=%.3f'%(error,f1score))
    plt.legend()

    #generating the code
    plt.show()


    #saving the file
    save_csv_table("molecular_motrs.csv",(target.numpy()[::J],rec.numpy()[::J])\
            ,labels="x,ini,rec", X=data.numpy()[::J])
