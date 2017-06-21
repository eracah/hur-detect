


import tensorflow as tf



def reshape_list(l, shape=None):
    """Reshape list of (list): 1D to 2D or the other way around.
    Args:
      l: List or List of list.
      shape: 1D or 2D shape.
    Return
      Reshaped list.
    """
    r = []
    if shape is None:
        # Flatten everything.
        for a in l:
            if isinstance(a, (list, tuple)):
                r = r + list(a)
            else:
                r.append(a)
    else:
        # Reshape to list of list.
        i = 0
        for s in shape:
            if s == 1:
                r.append(l[i])
            else:
                r.append(l[i:i+s])
            i += s
    return r



def get_int_tensor_shape(tensor):
    return convert_tf_shape_to_int_tuple(tensor.get_shape())

def convert_tf_shape_to_int_tuple(tf_shape):
    return tuple([dim.value for dim in tf_shape])

def shape_key(tensor):
    return get_int_tensor_shape(tensor)

def sort_some_lists_of_tensors(*lists):
    #print lists
    
    ret = [sorted(tensor_list,key=shape_key, reverse=True) for tensor_list in lists]
    if len(lists) == 1:
        ret = ret[0]
    return ret
    
    
    





