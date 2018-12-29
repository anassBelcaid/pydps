"""
Dps Filter in one dimension. The filter scan each edge to decide the existence
or absence of a discontinuity
"""

from linalg import gemanPriorMatrix
import torch
import numpy as np
from helper_functions import non_maxima_supression_
from datasets.datasets import load_random_dataset

class DpsFilter1D(object):
    """DPs Filter in one dimension, the filter scan each edge by solving a
    tridiagonal systme and decide by a threshold the existence or absence of a
    discontinuity
    """
    def __init__(self, lam, window):
        """
        Constructor with a given smoothness power and window
        :lam:  lambda is the smoothness power and also represent the lenght of
        each plateau
        : window: lenght of the window at one side (.e.g window=1 ==> will
        consider an array of 2*(window+1) on each edge
        """

        self.lam = lam
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

        
    
    def __call__(self, signal):
        """Method to filter a signal an return the line process

        :signal:  initial signal a torch tensor
        :returns: gradient position makred by 1
        """

        #preparing the second member
        b= self._second_member(signal)

        #solving the each position
        sol = torch.potrs(b,self.A_decom,upper=False)

        #getting the imporant information
        left  = sol[0,:-self.window]
        right = sol[-1,self.window:]

        #gradi
        gradient  = torch.abs(right[1:]-left[:-1])

        #threshold
        torch.threshold_(gradient,0.3,0)

        #non maxima supression
        line_process = non_maxima_supression_(gradient)

        return line_process, gradient,(left,right)
    
    

if __name__ == "__main__":
    fil  = DpsFilter1D(10,2)

    #loading a data set
    data,target = load_random_dataset(5,200)

    line_process, gradient,(left,right) = fil(data[0])
    print(line_process)






        
