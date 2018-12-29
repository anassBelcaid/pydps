import numpy as np
import torch as tc

def random(size,num_dis,geom_prob=0.05,min_dist=20,min_jump=0.2):
    """Generate a random signal of signal size, discontinuities positions
    follows a geom probability

    :size:  size of the signal
    :num_dis: number of discontinuities
    :geom_prob: geometric law parameter (\lambda)
    : min_dist: minimal distance between Two jumps
    :min_jump:  minimal jump for each discontinuit
    :returns: torch tensor containing the 
    """
    dis = [] 
    jumps = 2*np.random.rand(num_dis)-1

    k=0       # protec with max iteration
    rem = size          # remaining size

    while(len(dis)<num_dis and k <1000):
        #generating a poisson trial
        l = np.random.geometric(geom_prob)

        if(l>min_dist and l<rem):
            if(len(dis)==0):
                dis.append(l)
            else:
                dis.append(l+dis[-1])
            rem -= l
        k +=1

    assert(len(dis) == num_dis), " Probability too low"

    

    return  _discriteSignal(size,dis,jumps)


def _discriteSignal(size,dis,jumps,initial_value=0):
    """ Function to create a PWC with given discontinuities and jumps

    :size: size of the signal
    :dis: Array of the jumps positions
    :jumps:  jumps values at each dicontinuity
    :initial_value: initial value at the first plateau
    :returns: torch tensor containing the pwc signal
    """

    assert(len(dis) == len(jumps)), " dis and jumps should have the same lenghts"

    signal = tc.zeros(size)
    #initial pleateau
    signal[:dis[0]] = initial_value

    value=initial_value

    for i in range(len(dis)-1):
        value += jumps[i]
        signal[dis[i]:dis[i+1]] = value
    
    #last plateau
    value += jumps[-1]
    signal[dis[-1]:] = value

    return signal


