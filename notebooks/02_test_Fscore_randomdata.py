from pydps.filter1d import DpsFilter1D
from pydps.datasets.datasets import load_random_dataset
import argparse
import matplotlib.pyplot as plt
import numpy as np


#default parser
parser = argparse.ArgumentParser()


parser.add_argument("-d","--device",type=str,choices=['cpu','cuda']\
        ,help="device to perform the computation",default='cpu')

parser.add_argument("-b", "--batchsize", default=100, help="size of the\
        batch ",type=int)
parser.add_argument("-s", "--sigsize", default=500, help="Size of each \
        signal in the dataset",type=int)
parser.add_argument("-l", "--lamda", default=20, help="smoothness power"\
        ,type=int)
parser.add_argument("-H","--sensitivity", default=0.2,type=float,\
        help='Sensisitivy of the filter')

parser.add_argument("-w", "--window", default=10, help="filter window"\
        ,type=int)
parser.add_argument("-v", "--verbosity", action="store_true", default=False,
        help="verbosity")
#parse arguments
args = parser.parse_args()



if __name__ == "__main__":

    fil  = DpsFilter1D(args.lamda,args.sensitivity,args.window)
    device = args.device

    if args.verbosity:
        print("using device %s"%device)
        print("batch size= %d"%args.batchsize)
        print("signal size= %s"%args.sigsize)
        print("filter smoothness= %d"%args.lamda)
        print("filter sensibility= %.2f"%args.sensitivity)
        print("filter window =%d"%args.window)

    #loading a data set
    data,target = load_random_dataset(args.batchsize,args.sigsize,0.1)


    #applying the filter on the dataset
    fscore, Statistics = fil.score(data,target,device=device)


    #printing the statistics
    print("mean F score is %.2f"%fscore)
    mean_time = np.mean(np.array(Statistics['time'][1:]))
    print("mean time is %.2e"%mean_time)




