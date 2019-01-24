from __future__ import print_function  # python 3.x version
import pickle                          
import numpy as np
import os
from scipy.misc import imread

def load_CIFAR_batch(filename):
  """ load single batch of cifar """
  with open(filename, 'rb') as f:
    datadict = pickle.load(f, encoding='latin1')
    X = datadict['data']
    Y = datadict['labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
    #print(X.shape) # (10000, 32, 32, 3)
    Y = np.array(Y)
    return X, Y
    

# ROOT : Cifar10의 데이터가 담긴 디렉토리
# 폴더의 내용물
# data_batch1
# data_batch2
# data_batch3 
# data_batch4
# data_batch5
# test_batch
def load_CIFAR10(ROOT):
  """ load all of cifar """
  xs = []
  ys = []
  for b in range(1,6):
    f = os.path.join(ROOT,'data_batch_%d' % (b, ))
    X, Y = load_CIFAR_batch(f)
    xs.append(X)
    ys.append(Y)
  # 10000개씩 뽑아내고 그것을 append
  
  Xtr = np.concatenate(xs)
  Ytr = np.concatenate(ys)
  del X, Y
  Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
  return Xtr, Ytr, Xte, Yte