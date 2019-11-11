"""
Test the timing of the filter to restore the molecular mouvement
with different size
"""


from pydps.datasets.datasets import load_molecular_motor
from pydps.filter1d import DpsFilter1D
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch

######################
#  Arguments parser  #
######################
#default parser
parser = argparse.ArgumentParser()
parser.add_argument("-l", "--lamda", default=20, help="smoothnes"\
        , type=int)
parser.add_argument("-H", "--sensitivity", default=0.2, type=float,\
        help='Sensisitivy of the filter')
parser.add_argument("-w", "--window", default=10, help="filter window"\
        , type=int)
parser.add_argument("-v", "--verbosity", action="store_true", default=False,\
        help="verbositiy")
parser.add_argument("-p", "--param", default='size', help="pamester of\
        the simulation", choices = ['size','window'])
parser.add_argument("--winMin", default=9, \
        help="minimal window for simulation",type=int)
parser.add_argument("--winMax", default=3000, \
        help="maximal window for simulation",type=int)
parser.add_argument("-s","--simSize", default=200, \
        help="Simuation size",type=int)
parser.add_argument("-d","--device", default='cpu',\
        choices =['cpu','cuda'],help='device to execute  code') 
#parse arguments
args = parser.parse_args()


if __name__ == "__main__":

    if args.verbosity:
        for arg in vars(args):
            print(arg,"= ", getattr(args,arg))

    #loading the data
    if(args.param == 'size'):

        X,y = load_molecular_motor(5000)
        X.squeeze_(), y.squeeze_()

        #windows values
        windows = torch.linspace(args.winMin, args.winMax, args.simSize).int()

        
        #simulation data
        data = np.zeros((args.simSize, 2))
        data[:,0]  = windows.numpy()

        for (i,w) in enumerate(windows.numpy()):

            if(args.verbosity and i%10):
                print("w = %d"%w)
            #Filter 
            filt = DpsFilter1D(args.lamda, args.sensitivity,w)

            #calling the filter
            line,grad,(left,right)= filt(X,device=args.device)

            data[i,1]=filt.time

            #poly fit

    
        C = np.polyfit(np.log(data[:,0]), np.log(data[:,1]), 1)
        print('C=',C)

        plt.loglog(data[1:,0],data[1:,1])
        plt.show()
    


    
