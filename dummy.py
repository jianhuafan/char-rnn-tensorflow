import tensorflow as tf

from tensorflow.contrib import quantize

x = tf.constant([0.3, 0, 2.3, 1.5, -4.6])
y = tf.round(x)
y = tf.clip_by_value(y, -1, 1)

with tf.Session() as session:
    # Equivalent to: a = np.array( [1, 2, 3, 1] )
    session.run( x )
    print( x.eval() )
    # Equivalent to: a[a==1] = 0
    session.run( y )
    print( y.eval() )
