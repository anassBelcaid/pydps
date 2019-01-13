######################################################
#  Assess the accuracy of the neuroblastoma dataset  #
######################################################

import numpy as np
import matplotlib.pyplot as plt
from pydps.filter1d import DpsFilter1D
from pydps.datasets.neuroblastoma import Neuroblastoma
import argparse
import torch
from sklearn.metrics import auc,confusion_matrix,f1_score
from sklearn.metrics import roc_curve



#default argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-w", "--window", default=30, \
        help="window  of the filter",type=int)

parser.add_argument("-l", "--lamda", default=100, \
        help="Smoothness power",type=int)
parser.add_argument("-H", "--sensitivity", default=0.4, \
        help="sensibility of the filter",type=float)

parser.add_argument("-s", "--simSize",default=20,\
        help='simulation size', type=int)
args = parser.parse_args()


if __name__ == "__main__":
    
    #loading the data
    data = Neuroblastoma()
    
    #creating the filter
    filt =  DpsFilter1D(args.lamda,args.sensitivity,args.window)


    finding = []
    choices = np.random.choice(range(1,len(data)),args.simSize,replace=False)
    for index in choices:
        print("recovering patient i=%3d"%index)
        #trying one example
        patient = data.patient_sample(index)


        for chrom in patient:
            sig, state = patient[chrom]

            #changing the state
            state = 1 if state=='breakpoint' else 0
            sig = sig.astype('double')

            #to tensor 
            sig = torch.from_numpy(sig)

            #recovering the signal
            line,grad,(left,right) = filt(sig)

            #state by line process
            state_dps = 1 if line.sum()>0 else 0

            #adding the finding
            finding.append((state,state_dps))
        


    #converting the finding
    finding = np.array(finding)
    y_true = finding[:,0]
    y_pred = finding[:,1]

    #printing the statistics
    print("f1 score is %.2f"%f1_score(y_true,y_pred))
    print("confusions matrix: \n ",confusion_matrix(y_true,y_pred))

    #auc   getting the false positive and true  positive rates
    fpr, tpr, thresholds  = roc_curve(y_true,y_pred,pos_label=1)
    print("auc score = %.2f"%auc(fpr,tpr))
        


