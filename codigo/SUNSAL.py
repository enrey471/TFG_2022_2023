import sys
import scipy as sp
import numpy as np
import scipy.linalg as splin
from numpy import linalg as LA

def sunsal(M, y, AL_iters = 100, tol = 1e-4, x0 = None):

    #--------------------------------------------------------------
    # test for number of required parameters
    #--------------------------------------------------------------
    [LM,p] = M.shape # mixing matrixsize
    [L,N] = y.shape # data set size
    if LM != L:
        sys.exit('mixing matrix M and data set y are inconsistent')

    #--------------------------------------------------------------
    # Read the optional parameters
    #--------------------------------------------------------------
    AL_iters = int(AL_iters)
    if (AL_iters < 0 ):
        sys.exit('AL_iters must a positive integer')

    # compute mean norm
    norm_m = splin.norm(M)*(25+p)/float(p)
    # rescale M and Y and lambda
    M = M/norm_m
    y = y/norm_m

    if x0 is not None:
        if (x0.shape[0]==p) or (x0.shape[0]==N):
            sys.exit('initial X is not inconsistent with M or Y')

    #---------------------------------------------
    #  Constants and initializations
    #---------------------------------------------
    mu = 0.01

    [UF,SF] = splin.svd(sp.dot(M.T,M))[:2]
    IF = sp.dot( sp.dot(UF,sp.diag(1./(SF+mu))) , UF.T )
    print(IF)
    Aux = (1./IF.sum()) * sp.sum(IF,axis=1,keepdims=True)
    x_aux = sp.sum(Aux,axis=1,keepdims=True)
    IF1 = IF - sp.dot(Aux,sp.sum(IF,axis=0,keepdims=True))


    yy = sp.dot(M.T,y)

    #---------------------------------------------
    #  Initializations
    #---------------------------------------------

    # no intial solution supplied
    if x0 is None:
       x = sp.dot( sp.dot(IF,M.T) , y)
    else:
        x = x0

    z = x
    # scaled Lagrange Multipliers
    d  = 0*z

    #---------------------------------------------
    #  AL iterations - main body
    #---------------------------------------------
    tol1 = sp.sqrt(N*p)*tol
    tol2 = sp.sqrt(N*p)*tol
    i=1
    res_p = sp.inf
    res_d = sp.inf
    maskz = sp.ones(z.shape)
    mu_changed = 0

    while (i <= AL_iters) and ((abs(res_p) > tol1) or (abs(res_d) > tol2)):
            print(x)
            # save z to be used later
            if (i%10) == 1:
                z0 = z
            # minimize with respect to z
            z = sp.maximum(x-d,0)
            # minimize with respect to x
            x = sp.dot(IF1,yy + mu*(z+d)) + x_aux
            # Lagrange multipliers update
            d -= (x-z)
            
            # update mu so to keep primal and dual residuals whithin a factor of 10
            if (i%10) == 1:
                # primal residue
                res_p = splin.norm(x-z)
                # dual residue
                res_d = mu*splin.norm(z-z0)
                # update mu
                if res_p > 10*res_d:
                    mu = mu*2
                    d = d/2
                    mu_changed = True
                elif res_d > 10*res_p:
                    mu = mu/2
                    d = d*2
                    mu_changed = True

                if  mu_changed:
                    # update IF and IF1
                    IF = sp.dot( sp.dot(UF,sp.diag(1./(SF+mu))) , UF.T )
                    Aux = (1./IF.sum()) * sp.sum(IF,axis=1,keepdims=True)
                    x_aux = sp.sum(Aux,axis=1,keepdims=True)
                    IF1 = IF - sp.dot(Aux,sp.sum(IF,axis=0,keepdims=True))
                    mu_changed = False

            i+=1

    return x, res_p, res_d, i

#--------------------------------------------------------------
# example
#--------------------------------------------------------------
print(sunsal(np.array([[2,1,4],[0,2,0],[1,1,0],[0,3,1],[1,5,0]]),np.array([[2,2],[3,3],[1,1],[3,3],[1,1]]), 9))
