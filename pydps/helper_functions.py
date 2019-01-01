"""
Set of helper function for the dps filter
"""
import torch
import numpy as np


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
    

def _pushWithMinimalDistance(Set,elem, min_dist):
    """
    Function to push an element in th Queue if it's 
    far away from the queue by at leat min_dist
    
    :Set: Elements of the set  ( list)
    :elem:  element to be added
    :min_dist: minimum distance between two elements in the set
    """
    
    #nothing to do the if the set is empty
    if(len(Set)==0):
        Set.append(elem)
    else:

        #finding the closest element in the set
        closest_dist = np.abs(np.array(Set)-elem).min()

        #adding the element if it's far from the rest of the lement
        if(closest_dist>=min_dist):
            Set.append(elem)



def priorityQueueWithMinimalDistance(signal,min_dist):
    """
    get the discontinuities positions by concidering a priority queue on the
    signal values, and pushing those values with a minimal distance between the 
    discontinuities in order to avoid several edges

    :signal: torch signal ( generally represent the gradient 
    :min_dist: minimal distance between two jumps

    :return: return line process with non discontinuities
    """
    
    #sorting the array by descending order
    indices = np.flip(np.argsort(signal.numpy()))


    #Constructing the queue
    L = []
    for index in indices:
        if(signal[index]):
            _pushWithMinimalDistance(L,index,min_dist)

    line_process = torch.zeros_like(signal,dtype=torch.int32)

    line_process[L] = 1
    line_process[:min_dist]=0
    line_process[-min_dist:]=0
    
    return line_process






if __name__ == "__main__":
    
    grad = torch.tensor([0,0,0.4,0.8,0.5,0,0,0])
    print(grad)

    line = priorityQueueWithMinimalDistance(grad,3)
    print(line)




    

    

