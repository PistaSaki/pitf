from pitf import nptf
import numpy as np
import tensorflow as tf

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

    
if __name__ == "__main__":
    test_reduce_all()
    test_reduce_any()
    test_reduce_prod()
    
    print("Tests passed.")
    
