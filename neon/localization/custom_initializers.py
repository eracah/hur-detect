__author__ = 'racah'
from neon.initializers.initializer import Initializer
import numpy as np
from operator import mul

class HeWeightInit(Initializer):
    """ class for initializing parameter tensors with values
    drawn from a zero mean normal distribution with a stdev of sqrt(2/fanin)
    as discussed in Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
    the bias is set to 0
    """
    def __init__(self, name='HeWeightInit'):
        super(HeWeightInit, self).__init__(name=name)

    def fill(self, param):
        #print param.shape
        param[:] = self.be.rng.normal(0.0, np.sqrt(2.0 / param.shape[0]), param.shape)


class AveragingFilterInit(Initializer):
    def __init__(self, name='AveragingFilterInit'):
        super(AveragingFilterInit, self).__init__(name=name)

    def fill(self, param):
        print param.shape
        size = reduce(mul, param.shape)
        param[:] = 1./ size * np.ones(param.shape)


