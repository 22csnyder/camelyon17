
import tensorflow as tf
import numpy as np


#logits:
a=0.6
x=1.4


def sigmoid(x):
    return (1+np.exp(-x))**-1

px=sigmoid(x)
py=a*px


#z=0.2
#def xe(p,z=z):
#    return -z*np.log(p)-(1-z)*np.log(1-p)
#
#def sxe(l,z=z):
#    return xe(sigmoid(l))
#
#def lsy(l,z=z,a=a):
#    return -z*np.log(a)-z*np.log(sigmoid(l))
#
#
#def lsy2(l,z=z,a=a):
#    return -z*np.log(1-a*sigmoid(l))
#
#def f(l,z=z,a=a):
#    return max(-l,0)+np.log(1+np.exp(-np.abs(x)))-np.log(a)


def x2y(x):#works
    return -np.log( (1/a)*(1+np.exp(-x)) - 1)


def safe_x2y(x):
    return -np.log( (1-a)/a*min(np.exp(x),1) + (1/a)*min(np.exp(-x),1) ) + min(x,0)


