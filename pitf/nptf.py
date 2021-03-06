"""
Contains functions working for both numpy and tensorflow objects.
They all have a direct numpy or tensorflow origin (usualy both). 
The only exceptions are:
    batched_gather_nd
    np_dtype
    logist
    log_odds
"""
import numpy as np
import tensorflow as tf
import numpy.linalg as la

from .ptf import is_tf_object, any_is_tf

def identity(x, name):
    if is_tf_object(x):
        return tf.identity(x, name = name)
    else:
        return x

def gather(a, indices, axis = 0, name = None):
    if any_is_tf(a, indices, axis):
        return tf.gather(a, indices = indices, axis = axis, name = name)
    else:
        return np.take(a, indices = indices, axis = axis)
        
def gather_nd(a, indices, name = None):
    if any_is_tf(a, indices):
        return tf.gather_nd(a, indices=indices, name = name)
    else:
        a = np.array(a)
        return a[tuple(np.rollaxis(np.array(indices), -1))]
                 
def batched_gather_nd(a, indices, batch_ndim):
    """Return tensor `t` s.t. `t[i] == gather_nd(a[i], indices[i])` for any multiindex `i` of length `batch_ndim`.
    
    Args:
        a: tensor (np or tf)
        indices: integer-valued tensor (np or tf)
        batch_ndim: int
        
    Notes:
        The first `batch_ndim` axes of `a` and `indices` should be broadcast-compatible.
    
    Returns:
        tensor `t` in numpy or tensorflow.
    
    """
    a_shape = shape(a)
    #print("a_shape ={}".format(a_shape))
    indices_ndim = ndim(indices)
    assert indices_ndim >= batch_ndim + 1, "indices_ndim = {} should be greater than batch_ndim = {}".format(
        indices_ndim, batch_ndim)
    
    batch_shape = broadcast_shape(a_shape[:batch_ndim], shape(indices)[:batch_ndim])
    #print(batch_shape)
    
    ## we force `indices` to have this batch_shape (needed for broadcasting in `indices` to work)
    indices = indices * ones(
        shape = concat([batch_shape, [1] * (indices_ndim - batch_ndim)] ), 
        dtype = indices.dtype
    )
    
    ###
    sh = concat([ 
            batch_shape, 
            shape(indices)[batch_ndim: -1], 
            [1] 
        ])
    #print(batch_shape.dtype, sh.dtype, indices.dtype)
    
    ## we wanto to apply gather_nd, for this we must add indices
    ## that traverse the first `batch_ndim` dimensions of our tensors
    ranges = [cast(arange(d), dtype = np_dtype(indices)) for d in unstack(sh)]
    #print("ranges shapes =",[t.shape for t in ranges])
    
    ## in order to broadcast in `a` we must slightly modify `ranges`:
    ranges = [
        minimum(r , d-1) 
        for r, d in zip(ranges[:batch_ndim], unstack(cast(a_shape, np_dtype(indices)))[:batch_ndim]) 
    ] + ranges[batch_ndim: ]
    #print("ranges =", ranges)
    augmented_indices = concat(
        tensors = meshgrid(*ranges, indexing = "ij")[:batch_ndim] + [indices],
        axis = -1
    )
    
    #print("augmented_indices.shape =", augmented_indices.shape)
    
    ## infer the static shape in tensorflow 
    if is_tf_object(augmented_indices):
        static_batch_shape = tf.broadcast_static_shape(
            static_tf_shape(a)[:batch_ndim], 
            static_tf_shape(indices)[:batch_ndim]
        )
        #print("static_batch_shape =", static_batch_shape)
        stat_aug_ind_shape = static_batch_shape.as_list() + indices.shape.as_list()[batch_ndim:]
        stat_aug_ind_shape[-1] += batch_ndim
        augmented_indices.set_shape(stat_aug_ind_shape)
        
        #print("augmented_indices.shape =", augmented_indices.shape)
    
        
    
    return gather_nd(a, augmented_indices)

def boolean_mask(tensor, mask, name = "boolean_mask", axis = 0):
    if any_is_tf(tensor, mask, axis):
        return tf.boolean_mask(tensor=tensor, mask=mask, name=name, axis=axis)
    else:
        a = np.array(tensor)
        axis = int(axis)
        assert -a.ndim < axis < a.ndim
        if axis < 0:
            axis += a.ndim
        selector = axis * (slice(None), ) + (mask, )
        return a[selector]
        
    


def broadcast_shape(shape_x, shape_y):
    if any_is_tf(shape_x, shape_y):
        #print("broadcasting tf shapes:{}, {}".format(shape_x, shape_y))
        return tf.broadcast_dynamic_shape(shape_x, shape_y)
    else:
        #print("broadcasting np shapes:{}, {}".format(shape_x, shape_y))
        for x, y in zip(shape_x, shape_y):
            assert x == y or x == 1 or y == 1, "Incompatible shapes {}, {}".format(shape_x, shape_y)
        return np.maximum(shape_x, shape_y) 

def meshgrid(*args, **kwargs):
    if any_is_tf(*args):
        return tf.meshgrid(*args, **kwargs)
    else:
        return np.meshgrid(*args, **kwargs)
        
def arange(*args, **kwargs):
    if any_is_tf(*args, *kwargs.values()):
        return tf.range(*args, **kwargs)
    else:
        return np.arange(*args, **kwargs)
        
def stack(tensors, axis = 0):
    if any([is_tf_object(t) for t in tensors]):
        return tf.stack(tensors, axis = axis)
    else:
        return np.stack(tensors, axis = axis)
        
def unstack(value, num = None, axis = 0):
    if any_is_tf(value, num, axis):
        return tf.unstack(value= value, num = num, axis= axis)
    else:
        return list(np.rollaxis(value, axis = -1))[:num]
        
def shape(x):
    if is_tf_object(x):
        return tf.shape(x)
    else:
        return np.array(np.shape(x), dtype = np.int32)

def static_tf_shape(x):
    if is_tf_object(x):
        return x.shape
    else:
        return tf.TensorShape(np.shape(x))

        
def ndim(x):
    if is_tf_object(x):
        return len(x.shape)
    else:
        return np.ndim(x)
        
def concat(tensors, axis = 0):
    if any([is_tf_object(t) for t in tensors]):
        return tf.concat(tensors, axis = axis)
    else:
        return np.concatenate(tensors, axis = axis)
        
def zeros(shape, dtype = np.float32):
    if is_tf_object(shape):
        return tf.zeros(shape, dtype=dtype)
    else:
        return np.zeros(shape, dtype=dtype)

def ones(shape, dtype = np.float32):
    if is_tf_object(shape):
        return tf.ones(shape, dtype=dtype)
    else:
        return np.ones(shape, dtype=dtype)


def zeros_like(x, dtype = None):
    if is_tf_object(x):
        return tf.zeros_like(x, dtype=dtype)
    else:
        return np.zeros_like(x, dtype=dtype)

def ones_like(x, dtype = None):
    if is_tf_object(x):
        return tf.ones_like(x, dtype=dtype)
    else:
        return np.ones_like(x, dtype=dtype)
            

def transpose(x, perm):
    if is_tf_object(x) or is_tf_object(perm):
        return tf.transpose(x, perm)
    else:
        return np.transpose(x, perm)
        

def reshape(x, shape, name = None):
    if is_tf_object(x) or is_tf_object(shape):
        return tf.reshape(x, shape, name = name)
    else:
        return np.reshape(x, shape)
    
def flatten(x, name = None):
    return reshape(x, [-1], name)
        
def matmul(a, b):
    if is_tf_object(a) or is_tf_object(b):
        return tf.matmul(a, b)
    else:
        return np.matmul(a, b)
        
def tensordot(a, b, axes):
    if any_is_tf(a, b, axes):
        return tf.tensordot(a, b, axes)
    else:
        return np.tensordot(a, b, axes)
           

def reduce_sum(x, axis = None, keepdims = False, name = None):
    if any_is_tf(x, axis):
        return tf.reduce_sum(x, axis = axis, keepdims = keepdims, name = name)
    else:
        return np.sum(x, axis = axis, keepdims = keepdims)
 
def reduce_mean(x, axis = None, keepdims = False, name = None):
    if any_is_tf(x, axis):
        return tf.reduce_mean(x, axis = axis, keepdims = keepdims, name = name)
    else:
        return np.mean(x, axis = axis, keepdims = keepdims)
        

def reduce_prod(x, axis = None, keepdims = False, name = None):
    if any_is_tf(x, axis):
        return tf.reduce_prod(x, axis = axis, keepdims = keepdims, name = name)
    else:
        return np.prod(x, axis = axis, keepdims = keepdims)

def reduce_all(x, axis = None, keepdims = False, name = None):
    if any_is_tf(x, axis):
        return tf.reduce_all(x, axis = axis, keepdims = keepdims, name = name)
    else:
        return np.all(x, axis = axis, keepdims = keepdims)

def reduce_any(x, axis = None, keepdims = False, name = None):
    if any_is_tf(x, axis):
        return tf.reduce_any(x, axis = axis, keepdims = keepdims, name = name)
    else:
        return np.any(x, axis = axis, keepdims = keepdims)

        
def cast(x, dtype):
    if is_tf_object(x):
        return tf.cast(x, dtype)
    else:
        return np.array(x, dtype = dtype)
        
def maximum(x, y):
    if any_is_tf(x,y):
        return tf.maximum(x, y)
    else:
        return np.maximum(x, y)
        
def minimum(x, y):
    if any_is_tf(x,y):
        return tf.minimum(x, y)
    else:
        return np.minimum(x, y)

def pad(x, paddings, mode = "constant"):
    if is_tf_object(x):
        return tf.pad(x, paddings, mode)
    else: 
        return np.pad(x, paddings, mode)
  
def expand_dims(x, axis):
    if any_is_tf(x, axis):
        return tf.expand_dims(x, axis = axis)
    else:
        return np.expand_dims(x, axis = axis)
        
def tile(x, reps):
    if any_is_tf(x, reps):
        return tf.tile(x, reps)
    else:
        return np.tile(x, reps)
        
        
def exp(x, name = None):
    if is_tf_object(x):
        return tf.exp(x, name = name)
    else:
        return np.exp(x)

def log(x, name = None):
    if is_tf_object(x):
        return tf.log(x, name = name)
    else:
        return np.log(x)

        
def det(x):
    if is_tf_object(x):
        return tf.matrix_determinant(x)
    else:
        return la.det(x)
        
def einsum(equation, *inputs):
    if any_is_tf(*inputs):
        inputs = [
            x if is_tf_object(x) else tf.constant(x)
            for x in inputs
        ]
        return tf.einsum(equation, *inputs)
    else:        
        return np.einsum(equation, *inputs)
    
def equal(x, y, name = None):
    if any_is_tf(x, y):
        return tf.equal(x, y, name)
    else:
        return x == y

def not_equal(x, y, name = None):
    if any_is_tf(x, y):
        return tf.not_equal(x, y, name)
    else:
        return x != y

def np_dtype(x):
    if is_tf_object(x):
        return np.dtype(x.dtype.as_numpy_dtype)
    else:
        return x.dtype
        
def cumsum(x, axis = 0, name = None):
    if any_is_tf(x, axis):
        return tf.cumsum(x, axis, name = name)
    else:
        return np.cumsum(x, axis)


def logist(x, name = None):
    return identity(1 / ( 1 + exp(-x)), name = name)

def log_odds(p, name = None):
    return log( p / (1 - p), name = name)

def size(x, name = None, out_type = np.int32):
    if is_tf_object(x):
        return tf.size(x, name =name, out_type = out_type)
    else:
        return np.size(x)

def sign(x, name = None):
    if is_tf_object(x):
        return tf.sign(x, name =name)
    else:
        return np.sign(x)


def abs(x, name = None):
    if is_tf_object(x):
        return tf.abs(x, name =name)
    else:
        return np.abs(x)

if __name__ == "__main__":
    print("Hello world!")
    

