"""
Function and helper for all the linear algebra operations
"""
import numpy as np
import torch
import scipy.linalg as linAlg


#{{{ Helper function to move from condensed to full
def from_condensed_to_full(mat):
        """
        Convert dense representation to full form

        :mat: numpy array in 
        :return: full matrix form
        """

        #getting the size
        n, m = mat.shape

        #constructing the full matrix
        M = np.diag(mat[0, :])
        for k in range(1,n):
                M += np.diag(mat[k, :m-k], k) + np.diag(mat[k,:m-k],-k)
        return M
#}}}
#{{{ Line process matrix
def gemanPriorMatrix(lam,size,factorized=True):
        """
        generate the banded compressed truncated matrix
        :param lam   : regularization
        :param size  : size of the matrix
        :return:  full matrix in torch form
        """
        lam2=lam*lam;
        Ab=np.zeros((2,size));
        Ab[0,0]=1+lam2; Ab[0,size-1]=1+lam2;
        Ab[0,1:-1]=1+2*lam2;
        Ab[1,:]=-lam2;

        Ab = torch.tensor(from_condensed_to_full(Ab))
        if(factorized):
            Ab = torch.cholesky(Ab,upper=False)
        return Ab
#}}}
#{{{ Illustrate potrs
def illustrate_potrs():
    """
    Function to illlustrate the use of method positive triangular solve
    """

    # gettting a factorization of a linear systm
    A = gemanPriorMatrix(10,10,factorized=False)
    print("initial matrix:\n",A)

    A_decom = gemanPriorMatrix(10,10,factorized =  True)
    print("decomposition with cholesky\n",A_decom)

    # a set of secon members
    b= torch.from_numpy(np.random.normal(scale=2,size=(10,7)))
    print("SEven second members: \n", b)
    

    #solving the system using potrs
    sol = torch.potrs(b,A_decom, upper=False)

    #cheching the solution

    res = torch.mm(A,sol)-b
    print("residual error:")
    print(torch.norm(res,dim=0))   #this should be close to zero to each matrix

#}}}
    


if __name__ == "__main__":
    illustrate_potrs()
