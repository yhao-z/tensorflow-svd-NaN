import tensorflow as tf
import numpy as np

M = np.load('NaN_matrix.npy')
print(tf.reduce_any(tf.math.is_nan(tf.abs(M))).numpy())

# tensorflow svd
[s,u,v] = tf.linalg.svd(M)

print(s.numpy()[0:9])
print(tf.reduce_any(tf.math.is_nan(s)).numpy())
print(tf.reduce_any(tf.math.is_nan(tf.abs(u))).numpy())
print(tf.reduce_any(tf.math.is_nan(tf.abs(v))).numpy())

# numpy svd
[u,s,v] = np.linalg.svd(M)

print(s[0:9])
print(np.isnan(s).any())
print(np.isnan(u).any())
print(np.isnan(v).any())
