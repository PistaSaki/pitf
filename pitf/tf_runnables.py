from . import ptf
import tensorflow as tf
from types import SimpleNamespace

class TF_Runnable:
    """
    The offsprings of `TF_Runnable` should implement the following methods:
        
        _get_data_to_execute(self): returns data to be executed by `tensorflow.Session`
        
        _make_copy_from_executed_data(self, data): this method receives argument
            `data = session.run(self._get_data_to_execute())`
            and should return the reconstructed object.
    
    """
    def _get_data_to_execute(self):
        raise NotImplementedError(
            str(self.__class__) + " does not implement methods of TF_Runnable."
        )
    
    def _make_copy_from_executed_data(self, data):
        raise NotImplementedError(
            str(self.__class__) + " does not implement methods of TF_Runnable."
        )
        
    def tf_eval(self, feed_dict, sess):
        return run_in_session(self, feed_dict, sess)

class Not_Runnable_As_TF_Runnable(TF_Runnable):
    def __init__(self, obj):
        self.obj = obj
        
    def _get_data_to_execute(self):
        return []
    
    def _make_copy_from_executed_data(self, data):
        return self.obj

class TF_Object_As_TF_Runnable(TF_Runnable):
    def __init__(self, obj):
        self.obj = obj
        
    def _get_data_to_execute(self):
        return self.obj
    
    def _make_copy_from_executed_data(self, data):
        return data
        
class List_As_TF_Runnable(TF_Runnable):
    def __init__(self, l):
        assert isinstance(l, list)
        self.l = [into_TF_Runnable(x) for x in l]
        
    def _get_data_to_execute(self):
        return [x._get_data_to_execute() for x in self.l]
    
    def _make_copy_from_executed_data(self, data):
        assert isinstance(data, list)
        return [ obj._make_copy_from_executed_data(dat) for obj, dat in zip(self.l, data)]

class Dict_As_TF_Runnable(TF_Runnable):
    def __init__(self, d):
        assert isinstance(d, dict)
        self.d = { k: into_TF_Runnable(v) for k, v in d.items() }
        
    def _get_data_to_execute(self):
        return {k: v._get_data_to_execute() for k, v in self.d.items() }
    
    def _make_copy_from_executed_data(self, data):
        assert isinstance(data, dict)
        return {key: obj._make_copy_from_executed_data(data[key]) for key, obj in self.d.items()}
        
        
def into_TF_Runnable(obj):
    if ptf.is_tf_object(obj):
        return TF_Object_As_TF_Runnable(obj)
    
    if isinstance(obj, TF_Runnable):
        return obj
        
    if isinstance(obj, list):
        return List_As_TF_Runnable(obj)
    
    if isinstance(obj, dict):
        return Dict_As_TF_Runnable(obj)
    
    return Not_Runnable_As_TF_Runnable(obj)

def private_key(key):
    return key[0] == "_"
    
class RunnableNamespace(SimpleNamespace, TF_Runnable):
    def _get_data_to_execute(self):
        return {
            k: into_TF_Runnable(v)._get_data_to_execute()
            for k, v in self.__dict__.items()
            if not private_key(k)
        }
    
    def _make_copy_from_executed_data(self, data):
        return SimpleNamespace(**{
            k: into_TF_Runnable(v)._make_copy_from_executed_data(data[k])
            for k, v in self.__dict__.items()
            if not private_key(k)
        })


def run_in_session( x, feed_dict = None, sess = None, ):
    """
    If `x` is a tf object, we run it. If not, we leave it.
    If it is a list or dict, we do this for all the object it containts.
    """
    if sess is None:
        sess = tf.get_default_session()
        

    runnable = into_TF_Runnable(x)
    
    executed_data = sess.run(runnable._get_data_to_execute(), feed_dict = feed_dict)
    
    return runnable._make_copy_from_executed_data(executed_data)
    
#with tf.Session() as sess:
#    print(
#        run_in_session([
#            tf.constant([4, 4]),
#            [1, 2, tf.constant([1, 2])],          
#            {"a": [10, 20], "b": tf.constant([30, 40])}
#        ])
#    )
    
        