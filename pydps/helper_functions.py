"""
Set of helper function for the dps filter
"""
import torch


def non_maxima_supression_(signal):
    """
    tool function to remove the non maxima from the signal
    """

    #edges
    line_process = torch.zeros_like(signal,dtype=torch.int32)
    N=signal.shape[0]


    #loop inside
    for i in range(1,N-1):
        if((signal[i]>signal[i-1]) and (signal[i]>signal[i+1])):
            line_process[i]=1

    return line_process
    



    

    

