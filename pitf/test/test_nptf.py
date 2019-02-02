from pitf import nptf
import numpy as np
import tensorflow as tf
from pitf.ptf import is_tf_object

def test_reduce_all():
    assert nptf.reduce_all([True, True]) == True
    with tf.Session():
        assert nptf.reduce_all(tf.constant([True, True])).eval() == True
        
def test_reduce_any():
    assert nptf.reduce_any([False, False]) == False
    with tf.Session():
        assert nptf.reduce_any(tf.constant([False, True])).eval() == True

def test_reduce_prod():
    assert nptf.reduce_prod([2, 3]) == 6
    with tf.Session():
        assert nptf.reduce_prod(tf.constant([2, 3])).eval() == 6
        
def _check_fun(fun, **kwargs):
    with tf.Session():
        kwargs_eval = {
            k: v.eval() if is_tf_object(v) else v 
            for k, v in kwargs.items()
        }
        assert np.allclose(fun(**kwargs).eval(), fun(**kwargs_eval))
        
def test_boolean_mask():
    a = np.arange(12).reshape([3, 4])
    
    _check_fun( 
        fun = nptf.boolean_mask,
        tensor = tf.constant(a),
        mask = tf.constant([True, False, True]),
    )
    
    _check_fun( 
        fun = nptf.boolean_mask,
        tensor = tf.constant(a),
        mask = tf.constant([False, True, False, True]),
        axis = 1
    )
    
    _check_fun( 
        fun = nptf.boolean_mask,
        tensor = tf.constant(a),
        mask = tf.constant(a%2==0),
    )

    
if __name__ == "__main__":
    test_reduce_all()
    test_reduce_any()
    test_reduce_prod()
    test_boolean_mask()
    
    print("Tests passed.")
    
