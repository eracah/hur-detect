


import sys
if __name__ == "__main__":
    sys.path.append("../../../")
from dotpy_src.postprocessing.utils import get_int_tensor_shape
import tensorflow as tf
slim=tf.contrib.slim
from dotpy_src.losses.utils import abs_smooth as smooth_L1





