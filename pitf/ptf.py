import tensorflow as tf
import numpy as np

from numbers import Number

from toolz import reduce

def logist(x):
    return tf.exp(x) / ( 1 + tf.exp(x))

def log_odds(p):
    return tf.log( p / (1 - p))
    
    
def SparseTensorValue_to_dense_np(t):
    t1 = np.zeros(t.dense_shape)
    for ii, val in zip(t.indices, t.values):
        t1[tuple(ii)] += val
        
    return t1
 
def sparse_sum(l):
    return reduce(
        lambda a, b: tf.sparse_add(a, b),
        l
    )
    
def nan_to_num(v):
    return tf.where( tf.is_nan(v), tf.zeros_like(v, dtype = v.dtype), v)
    
def add_column_of_ones(table):
    return tf.concat([table, tf.ones_like(table[...,:1])], axis = 1)
    
def is_tf_object(x):
    return isinstance(x, (tf.Tensor, tf.Operation, tf.Variable))
    
def any_is_tf(*args):
    return any([is_tf_object(x) for x in args])
    
######## 
## Assigning to variables

class EmptyClass:
    pass
    
def add_assigner(x):
    if not hasattr(x, "ptf"):
        x.ptf = EmptyClass()
        
    if not ( hasattr(x.ptf, "placeholder") and  hasattr(x.ptf, "assign")):
        name = x.name.split(":")[0]
        validate_shape = all([dim.value is not None for dim in x.shape])
    
        x.ptf.placeholder = tf.placeholder(dtype = x.dtype, shape = x.get_shape(), 
                name = r"ptf/assign/placeholder/" + name)
        x.ptf.assign = tf.assign(x, x.ptf.placeholder, 
                name = "ptf/assign/" + name, validate_shape=validate_shape)
        
def assign_to(x, val, sess = None):
    if sess is None:
        sess = tf.get_default_session()
        
    add_assigner(x)
    
    sess.run(x.ptf.assign, {x.ptf.placeholder: val})
    
class SparseMatrix:
    def __init__(self, indices, values, dense_shape, name = None):
        self.indices = tf.Variable(indices, validate_shape=False, dtype = np.int64)
        self.indices.set_shape([None, 2])
        
        self.values = tf.Variable(values, validate_shape=False)
        self.values.set_shape([None])
        
        self.dense_shape = tf.Variable(dense_shape, dtype=np.int64)
        
        self.tensor = tf.SparseTensor( 
            indices= self.indices, values=self.values, dense_shape=self.dense_shape
        )
        
    def set_value(self, indices, values, dense_shape, sess = None):
        assign_to(self.indices, indices, sess)
        assign_to(self.values, values, sess)
        assign_to(self.dense_shape, dense_shape, sess)
        
        
############
## solving equations

def get_tf_step_of_Newton(x, f, df = None, w = None,  lam_cutoff = 0.01, clip_norm = 0.2):
    """
    Consider $F_k: R^n \to R^m$. We want to solve $F_k(x_k) = 0$ for all k.
    Thus 
    * x is a tf.Variable of shape (?, n)
    * f is a tensor of shape (?, m)
    * df is a tensor of shape (?, m, n)
    We do it for all k simultaneously, because tensorflow can paralelize it so it becomes much faster.
    In the rest of this explanation, assume there is only one fixed k and we no longer write it.
    
    So we have 
    $ F:R^n \to R^m$ and we are solving $ F(x) = 0 $. That means 
    $ F_j(x_0, ... , x_n) = 0 $ for all j = 0, ... m.
    
    The Newton method linearizes 
    $$ F(x + dx) = F(x) + A dx $$
    and then finds $dx$ from linear equa
    $$ A dx = b $$
    with $A = dF(x)$ and $b = -F(x)$.
    
    The linear equation is in general not well conditioned so we solve it as
    $$ dx = pinv(A) b $$
    where pinv(A) is the pseudo-inverse of A. This finds 
    
    In order not to jump too far we limit the $L_2$ norm of $dx$ to be less or equal to our clip_norm parameter.
    
    Actually we rather use what I call a "truncated pseudo inverse". That is, we do a SVD decomposition:
    $$ A = U S V $$
    with $S$ positive diagonal and $U, V$ orthogonal matrices.
    The usual pseudo-inverse is
    $$ pinv(A) = V^T S_{inv} U^T $$ 
    Here $S_{inv}$ is again diagonal with diagonal entry $\lambda$ of $S$ corresponding to 
    a diagonal entry $1 / \lambda$ if $ \lambda > \eps $ and 0 otherwise.
    
    We take instead $1 / \lambda$ if $\lambda$ > lam_cutoff, otherwise we take $ 1/ lam_cutoff$.
    
    The last parameter to explain is the weight vector w. Imagine that for some index k we don't have 
    to satisfy the equation $ F_j(x_0, ... , x_n) = 0 $ for all j = 0, ... m. 
    We accomplish this by setting w_j = 1 where we want to satisfy the equa and w_j = 0 where we do not.
    More generally, we can take w_j to be the weight of how strong we want to satisfy the equa $F_j(x) = 0$.
    Thus w is a tensor of shape (?, m).
    
    There is a more conceptual way of thinking of w. 
    Even without it, we can not in general satisfy all equas $F_j(x) = 0$ and correspondingly the equa $A dx = b$.
    The pseudo-inverse solution finds such dx that $A dx$ is closest possible to $b$ 
    with respect to the standard $L_2$ norm of $R^m$. When $w$ is given we don't use the standard $L_2$ norm 
    (given by the identity matrix) but the norm given by the diagonal matrix diag(w**2).
    Thus a possible generalization is to use a general positive-semidefinite matrix for scalar product in $R_m$.
    
    """
    
    
    x_dim, f_dim = x.shape[-1], f.shape[-1]
    
    if df is None:
        df = tf.stack( [tf.gradients(f[..., i], x)[0] for i in range(f_dim) ] , axis=-2)
        
    A = df
    b = -f

    if w is not None:
        A = w[..., None] * A
        b = w * b

    s, u, v_t = tf.svd(A)

    ## truncated inverse of s
    s_inv = tf.where(
        condition = tf.less_equal(s, lam_cutoff),
        x = tf.ones_like(s) / lam_cutoff,
        y = 1 / s    
    )

    delta_x_unclipped = tf.matmul(
        a = v_t,
        b = s_inv[..., None] * tf.matmul( u, b[..., None], transpose_a=True),
    )[..., 0]

    delta_x = tf.clip_by_norm(delta_x_unclipped, clip_norm=0.2, axes = -1)

    train_step = tf.assign_add(x, delta_x, name = "train_step")
    
    return train_step
        


#########################################
## finding index of a bin containing `t`

def get_bin_index(bins, t, name = "bin_index"):
    """
    returns (scalar) index of bin containing t.
    `t` is scalar, 
    `bins` is ordered 1D tensor
    """
    return tf.reduce_max(
        tf.where( 
            condition = tf.less_equal(bins, t), 
        ),
        name = name,
    )

def get_bin_indices(bins, tt, name = "bin_indices"):
    """
    the same as `get_bin_index`, only `tt` is 1D tensor 
    and the returned value is also 1D tensor.
    """
    return tf.map_fn(
        fn = lambda t: get_bin_index(bins = bins, t = t), 
        elems = tt,
        dtype = np.int64,
        name = name,
    )
  
#######################################################
## From TF tensors to NP arrays and back

def unstack_to_array(x, ndim = None, start_index = 0):
    """
    Returns a numpy array of tensorflow tensors 
    corresponding to successive unstacking of the first `ndim` 
    dimensions of `x` starting at `start_index`.
    If `ndim` is not specified, all the dimensions are unstack.
    """
    
    ## fussing about the shapes
    if x.shape == None:
        raise ValueError("In order to unstack a tensor you need to know its shape at least partially.")

    if ndim is None:
        end_index = len(x.shape)
        ndim = end_index - start_index
    else:
        end_index = start_index + ndim
    
    try:
        np_shape = [int(dim) for dim in x.shape[start_index:end_index]]
    except TypeError as e:
        raise ValueError(
            "The dimensions you want to unstack must be known in advance. " +
            "x.shape = " + str(x.shape) + " and you want to unstack dimensions " +
            str(start_index) + " to " + str(end_index) + "."
        ) from e
    
    tf_shape = tf.shape(x)
    
    ## reshape `x` so that all the unstack indices become one at position `start_index`
    xr = tf.reshape(
            tensor = x, 
            shape = tf.concat(
                values = [
                    tf_shape[:start_index],
                    np.array([np.prod(np_shape)], dtype=np.int32), 
                    tf_shape[end_index:]
                ],
            axis = 0
        )
    )
    
    ## unstack this one dimension into list
    l = tf.unstack(xr, axis = start_index)
    
    ## reshape this one dimensional list into a np.array of required shape
    return np.array(l).reshape(np_shape)
    
def stack_from_array(a, start_index = None, val_shape = None, dtype = None):
    """
    We assume `a` contains tf tensors of same shape (maybe scalars) and we stack them together.
    The indices of `a` will correspond to a group of indices 
    of the result starting at `start_index`.
    """
    a_flat = list(a.flat)
    if val_shape is None:
        try:
            val_shape = tf.shape(a_flat[0])
        except ValueError as exc:
            raise ValueError("val_shape can not be inferred.") from exc
            
    else:
        if any([ isinstance(x, Number) for x in a_flat]):
            # we replace by numerical zeros by tensors so that we can stack
            assert dtype is not None, (
               "If you want to automatically replace numbers, " +
               "please provide a dtype."
            )
            zeros = tf.zeros(shape=val_shape, dtype=dtype)
            a_flat = [ 
                x if not isinstance(x, Number) else x + zeros 
                for x in a_flat
            ]
            
    
    if start_index is None:
        start_index = 0
#    if start_index < 0:
#        start_index = val_ndim + start_index
        
#    val_ndim = len(val_shape)
#    assert 0 <= start_index <= val_ndim, (
#        "Problem: start_index = {};  val_ndim = {}.".format(start_index, val_ndim)
#    )

            
#    if start_index is None:
#        start_index = a.ndim 
#    if start_index < 0:
#        start_index = a.ndim + start_index
#    assert 0 <= start_index <= a.ndim, (
#        "Problem: start_index = {};  a.ndim = {}.".format(start_index, a.ndim)
#    )
  
    
    
    
    res = tf.stack(a_flat, axis = start_index)
    res = tf.reshape( res, 
        shape =  tf.concat(
            values = [
                val_shape[:start_index],
                a.shape,
                val_shape[start_index:]
            ],
            axis = 0
        )
    )
    return res
    
def array_to_tf(a):
    """
    Converts np.array `a` to one TensorFlow tensor.
    If `a` is numeric then we return tf.constant.
    If not, we assume it contains tf tensors of same shape (maybe scalars) and we stack them together.
    """
    if np.issubdtype(a.dtype, np.number):
        return tf.constant(a)
    else:
        return stack_from_array(a, start_index = 0)
        
def stack_from_array__keep_values_together(a, start_index = None):
    """
    We assume `a` contains tf tensors of same shape (maybe scalars) and we stack them together.
    The dimensions of the elements of `a` will correspond to a group of indices 
    of the result starting at `start_index`.
    """
    val_shape = tf.shape(a.flat[0])
    
    res = tf.stack(list(a.flat), axis = 0)
    res = tf.reshape( res, 
        shape =  tf.concat(
            values = [
                a.shape,
                val_shape,
            ],
            axis = 0
        )
    )
    
    if start_index is not None:
                     
        assert 0 <= start_index <= a.ndim
        val_ndim = tf.shape(val_shape)[0]
        res_ndim =  a.ndim + val_ndim # = tf.shape(tf.shape(res))[0] 
        res = tf.transpose( res,
            perm = tf.concat(
                values = [
                    tf.range(start_index),
                    tf.range(a.ndim, res_ndim),
                    tf.range(start_index, a.ndim)
                ],
                axis = 0
            )
        )
    return res