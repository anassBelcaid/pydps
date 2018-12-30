"""
Dps Filter in one dimension. The filter scan each edge to decide the existence
or absence of a discontinuity
"""

from pydps.linalg import gemanPriorMatrix
import torch
import numpy as np
from pydps.helper_functions import non_maxima_supression_
from pydps.datasets.datasets import load_random_dataset
from sklearn.metrics import precision_score,recall_score,f1_score
from time import time


#supperted devices
supported_devices = ['cpu','cuda']

class DpsFilter1D(object):

    """DPs Filter in one dimension, the filter scan each edge by solving a
    tridiagonal systme and decide by a threshold the existence or absence of a
    discontinuity
    """
    def __init__(self, lam,sensitivity, window):
        """
        Constructor with a given smoothness power and window
        :lam:  lambda is the smoothness power and also represent the lenght of
        each plateau
        :sensitivity: threshold on the gradient to detect a discontinuity
        : window: lenght of the window at one side (.e.g window=1 ==> will
        consider an array of 2*(window+1) on each edge
        """

        self.lam = lam
        self.sensitivity = sensitivity
        self.window = window
        
        #computing the decomposition on a one Geman matrix

        self.A_decom = gemanPriorMatrix(lam,window+1,factorized=True)

    def _second_member(self,signal):
        """
        function to create the second member for each edge
        from the tensor signal

        :singal: signal to be processed (N,)
        :return: tensor of size (2*(window+1), N+W), where each col is the
        signal associated to position the window in a given position of the
        signal
        """
        
        #padding the signal
        padded = torch.from_numpy(np.pad(signal.numpy(),self.window,'edge'))

        #getting the signal size
        N = signal.shape[0] + self.window
        part_size = self.window+1 # size of each part

        #preparing the tensor

        b = torch.zeros((part_size, N),dtype=torch.float64)

        for i in range(N):
            b[:,i] = padded[i:i+part_size]

        return b

        
    
    def __call__(self, signal,device='cpu'):
        """Method to filter a signal an return the line process

        :signal:  initial signal a torch tensor
        :returns: gradient position makred by 1
        """
        
        assert(device in supported_devices) , "unkown device %s "%device

        #preparing the second member
        b= self._second_member(signal)

        #solving the each position
        if(device=='cpu'):
            t=time()
            sol = torch.potrs(b,self.A_decom,upper=False)
            self.time=time()-t
        else:
            b= b.to('cuda')
            A_decom = self.A_decom.to('cuda')
            t=time()
            sol = torch.Tensor.potrs(b,A_decom,upper=False)
            self.time=time()-t

        #converting to memory in case of gpu
        if(device =='cuda'):
            sol = sol.cpu()

        #getting the imporant information
        self.left  = sol[0,:-self.window]
        self.right = sol[-1,self.window:]

        #gradi
        self.gradient  = torch.abs(self.right[1:]-self.left[:-1])

        #threshold
        torch.threshold_(self.gradient,self.sensitivity,0)

        #non maxima supression
        self.line_process = non_maxima_supression_(self.gradient)


        print("filtering took %.2e second"%self.time)

        return self.line_process, self.gradient,(self.left,self.right)

    def statistics_(self, gt):
        """Function to get the classification statistics from the ground truth

        :gt:  Ground truth
        :returns: Store the statistics on the dictionnary self.statistics

        """


        assert(self.line_process is not None), " You should compute edge before \
                running the statistics"

        self.statistics = {}

        #line process of the ground truth
        y_true  =  (gt[1:] != gt[:-1]).numpy()
        y_pred  =  self.line_process.numpy()
        
        #compute the statistics
        self.statistics['precision'] = precision_score(y_true,y_pred)
        self.statistics['recall'] = recall_score(y_true,y_pred)
        self.statistics['fscore'] = f1_score(y_true,y_pred)


        return self.statistics
    
    

if __name__ == "__main__":
    fil  = DpsFilter1D(200,0.25,200)
    device = 'cpu'
    print("using device %s"%device)

    #loading a data set
    data,target = load_random_dataset(10,10000)

    for i in range(5):
        
        #filtering  
        line_process, gradient,(left,right) = fil(data[i],device=device)


        #computing the statistics
        stats = fil.statistics_(target[i])
        print(stats)






        
