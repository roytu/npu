
import numpy as np

ig = 0.0
im = 0.5
iw = 0.88

isg = 0.2
ism = 0.2
isw = 0.2

def initialization(output_dim, input_dim=0, activationrange=2):
    """ glorot initialization. mean and stddev for backwards compatibility with truncated normal
    """
    return np.random.uniform(
        low=-np.sqrt(activationrange / (output_dim + input_dim)),
        high=np.sqrt(activationrange / (output_dim + input_dim)),
        size=dimension
    ).astype('float32')

def g_initialization(dimension, activationrange=2):
    return np.random.normal(
        loc=ig,
        scale=isg,
        size=dimension).astype('float32')

def w_initialization(dimension, activationrange=2):
    return np.random.normal(
        loc=iw,
        scale=isw,
        size=dimension).astype('float32')

def m_initialization(dimension, activationrange=2):
    return np.random.normal(
        loc=im,
        scale=ism,
        size=dimension).astype('float32')

