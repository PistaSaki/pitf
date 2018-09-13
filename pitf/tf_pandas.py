import tensorflow as tf
import numpy as np
from pandas import DataFrame
from collections import OrderedDict

from . import tf_runnables_02 as tf_runnables
from . import ptf

def variable_from_series(
    s, name = None, dtype = None, display_head= True, sess = None
):
    if dtype is None:
        dtype = s.dtype
        
    if name is None:
        name = s.name

    var = tf.Variable(
        initial_value= np.array(s, dtype = dtype), 
        dtype= dtype, 
        name = name,
        validate_shape = False,
    )

    var.set_shape([None])
    
    if sess is None:
        sess = tf.get_default_session()
        
    sess.run(var.initializer)
    
    if display_head:
        print(
            name,":", sess.run(var)[:5]
        )
    
    return var

class TFDF(tf_runnables.TF_Runnable):
    def __init__(
        self, *, 
        dic = OrderedDict(), df = None, 
        replace_float = False, sess = None, 
        name = None
    ):
        """
        Args:
            replace_float: If you want to replace all the float dtypes by one specified,
                you should provide it as this argument.
            sess: instance of tf.Session. If provided, we run the initializers.
        """
        
        self.name = name
        
        self._dic = OrderedDict()
        for key, value in dic.items():
            self[key] = value
            
        
        
        if df is not None:
    
            for c in df.columns:
                dtype = df[c].dtype
                if replace_float and dtype.kind == "f":
                    dtype = replace_float
                    
                self._dic[c] = variable_from_series(
                    s = df[c], 
                    name = self._tf_name_format(c), 
                    dtype = dtype, 
                    display_head = False, 
                    sess = sess
                )
                
    def _tf_name_format(self, column):
        if self.name:
            return "{}/{}".format(self.name, column)
        else:
            return column
            
            
    def __getitem__(self, key):
        return self._dic[key]
    
    def __setitem__(self, key, value):
        self._dic[key] = tf.identity(value, self._tf_name_format(key))
    
    def __getattr__(self, name):
        if not name in self._dic:
            raise AttributeError("There is no column '{}'.".format(name))
            
        return self._dic[name]
            
    def _get_data_to_execute(self):
        """Method of `TF_Runnable`.
        """
        return self._dic
    
    def _make_copy_from_executed_data(self, data):
        """Method of `TF_Runnable`.
        """
        return DataFrame(data)
    
    def feed_dict(self, df, check = True):
        """Return feed_dict feeding the data form DataFrame `df`.
        
        Args:
            df: `pandas.DataFrame` with the same columns as `self`.
            check: boolean indicating whether we should raise an exception
                when `df` does not contain all the columns in `self`.
        """
        if check:
            assert set(self._dic.keys()).issubset(df.columns), (
                "DataFrame df does not contain all the columns you want to set. "
            )
        cols = set(df.columns).intersection(self._dic.keys())
        return {
            self[c]: np.array(df[c])
            for c in cols
        }
    
    def assign(self, df, sess = None):
        """Assign the data form DataFrame `df` to the corresponding tensorflow variables.
        
        Args:
            df: `pandas.DataFrame` with the same columns as `self`.
        """
        for var, val in self.feed_dict(df).items():
            ptf.assign_to(var, val, sess = sess)
    
        