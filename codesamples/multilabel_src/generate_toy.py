'''
Created on Sep 5, 2013
generate toy data
TODO: generate data with correct correlation and sparsity properties

@author: Sanmi Koyejo; sanmi.k@gmail.com
'''

from sklearn.datasets import make_multilabel_classification
from sklearn.cross_validation import train_test_split
from utils import inverter
import numpy as np
import os

def generate_data(resultdir, N=100, D=30, K=10, ka=3):
    '''
    Generate data and save it
    N = # examples
    D = dims
    K = # classes
    ka = avg # classes
    '''
    
    X, Zc = make_multilabel_classification(n_samples=N, n_features=D, n_classes=K, n_labels=ka,\
                                           allow_unlabeled=False, random_state=1)
    # save results
    Z = inverter(Zc, K)
    index = np.arange(N, dtype=int)
    train = np.ones_like(index)
    testid = np.random.permutation(N)[:.2*N]
    train[testid] = 0
    test = np.logical_not(train)
    writefile = os.path.join(resultdir, "indata")
    
    try: os.makedirs(resultdir)
    except: pass
    
    np.savez(writefile, X=X, Z=Z, train=train, test=test)

if __name__ == "__main__":
    resultdir=os.path.expanduser(os.path.join("~", "workdir", "data", "multilabel", "toy"))
    generate_data(resultdir) 