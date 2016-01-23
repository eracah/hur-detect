__author__ = 'racah'
from neon import NervanaObject
import numpy as np
from neon.transforms.cost import Cost
class MeanCrossEntropyBinary(Cost):

    """
    Applies the binary cross entropy function

    Note:
        bprop assumes that shortcut is used to calculate derivative
    """

    def __init__(self, scale=1):
        """
        Initialize the binary cross entropy function

        Args:
            scale (float): amount by which to scale the backpropagated error
        """
        self.scale = scale

    def __call__(self, y, t):
        """
        Applies the binary cross entropy cost function

        Args:
            y (Tensor or OpTree): Output of previous layer or model
            t (Tensor or OpTree): True targets corresponding to y

        Returns:
            OpTree: Returns the binary cross entropy cost
        """
        a = - self.be.safelog(y) * t
        b = - self.be.safelog(1 - y) * (1 - t)
        return self.be.mean(a + b, axis=0)

    def bprop(self, y, t):
        """
        Computes the shortcut derivative of the binary cross entropy cost function

        Args:
            y (Tensor or OpTree): Output of previous layer or model
            t (Tensor or OpTree): True targets corresponding to y

        Returns:
            OpTree: Returns the (mean) shortcut derivative of the binary entropy
                    cost function ``(y - t) / y.shape[1]``
        """
        return self.scale * (y - t) / y.shape[0]