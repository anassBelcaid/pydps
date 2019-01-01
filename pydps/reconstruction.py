"""
Module to reconstruct a signal either by it' snoised version or given it's
contour
"""


import numpy as np
import torch
from pydps.linalg import gemanPriorMatrix
import matplotlib.pyplot as plt



def blocks_from_edge1d(edge):
    """
    function to get the blocks from a line process (edge) in one dimension
    
    :edge: torch array line process 0 in continuities and 1 otherwise
    :return: A numpy array 2d dimension [ (l_1,r_1), ......, (l_n,r_n)]
    where each tuple are the extremities of on block 
    """

    L = np.nonzero(edge.numpy())[0]

    blocks = np.zeros((len(L)+1,2),dtype=np.int32)
    
    #left parts
    blocks[1:,0]=L; blocks[0,0]=0

    #right parts
    blocks[:-1,1]=L; blocks[-1,1]=len(edge)+1


    return blocks


def reconstruct_mean_from_edge1d(signal,edge):
    """Function to replace each block by its mean given the contours

    :signal: torch tensor 1D (dim N)  noised version of the signal
    :edge: torch tensor 1D (dim N-1) edges for the signal
    :returns: recoverd signal by replacing each block by it's mean
    """

    #getting the blocks
    blocks = blocks_from_edge1d(edge)


    #recovered signal
    recovered = torch.zeros_like(signal)

    for (l,r) in blocks:
        recovered.numpy()[l:r] = signal[l:r].numpy().mean()

    return recovered

def reconstruct_median_from_edge1d(signal,edge):
    """Function to replace each block by its mean given the contours

    :signal: torch tensor 1D (dim N)  noised version of the signal
    :edge: torch tensor 1D (dim N-1) edges for the signal
    :returns: recoverd signal by replacing each block by it's mean
    """

    #getting the blocks
    blocks = blocks_from_edge1d(edge)


    #recovered signal
    recovered = torch.zeros_like(signal)

    for (l,r) in blocks:
        recovered.numpy()[l:r] = np.median(signal[l:r].numpy())

    return recovered



def _reconstruct_geman(signal,lam):
    """ Reconstruct the signal by considering the continuous geman model
    :signal: torch tensor 
    :lam : Smoothness power
    :returns: solution to the the (Ax= signal) where A is the continuous matrix
    """

    A =gemanPriorMatrix(lam,len(signal),factorized=True)
    

    return torch.potrs(signal,A.float(),upper=False)
    

def reconstruct_geman_from_edge1d(signal,edge,lam):
    """Function to replace each block by its mean given the contours

    :signal: torch tensor 1D (dim N)  noised version of the signal
    :edge: torch tensor 1D (dim N-1) edges for the signal
    :returns: recoverd signal by replacing each block by it's mean
    """

    #getting the blocks
    blocks = blocks_from_edge1d(edge)


    #recovered signal
    recovered = torch.zeros_like(signal)
    for (l,r) in blocks:
        recovered[l:r] = _reconstruct_geman(signal[l:r],lam).squeeze()

    return recovered
if __name__ == "__main__":

    lam =3 
    signal  =   torch.randn(12)
    edge   = torch.tensor([0,0,0,1,0,0,0,1,0,0,0])

    recovered  = reconstruct_geman_from_edge1d(signal,edge,lam) 
    plt.plot(signal.numpy())
    plt.stem(recovered.numpy())
    plt.show()




