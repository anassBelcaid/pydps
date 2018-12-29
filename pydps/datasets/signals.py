import numpy as np
import torch as tc

def random(size,num_dis,min_jump=0.3):
    """Generate a random signal of signal size, discontinuities positions
    follows a geom probability

    :size:  size of the signal
    :num_dis: number of discontinuities
    :min_jump: minimu jump
    :returns: torch tensor containing the 
    """

    # array of the discontinuities
    dis = sorted(np.random.choice(range(1,size),num_dis,replace=False))
    # TODO : Should add min jumps to avoid 
    jumps = min_jump + np.random.rand(num_dis)


    #flipping given jumps
    mask = np.random.choice([True,False],num_dis,replace=True)
    jumps[mask] = -jumps[mask]

    return _discriteSignal(size,dis,jumps)







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



if __name__ == "__main__":

    X = random(100,5)

    print(X)
    
