import tensorflow as tf
import numpy as np

M = np.load('NaN_matrix.npy')
print(tf.reduce_any(tf.math.is_nan(tf.abs(M))).numpy()) # False, no nan in M

# tensorflow svd
[s,u,v] = tf.linalg.svd(M)

print(s.numpy()[0:9]) # [nan 0.01783315 0.00682789 0.00398225 0.00252413 0.0014872 0.00082805 0.00066117 0.00051815]
print(tf.reduce_any(tf.math.is_nan(s)).numpy())         # True, nan in s
print(tf.reduce_any(tf.math.is_nan(tf.abs(u))).numpy()) # True, nan in u
print(tf.reduce_any(tf.math.is_nan(tf.abs(v))).numpy()) # True, nan in v

# numpy svd
[u,s,v] = np.linalg.svd(M)

print(s[0:9]) # [0.01783314 0.00682789 0.00398224 0.00252413 0.0014872 0.00082805 0.00066117 0.00051815 0.00050564]
print(np.isnan(s).any()) # False, no nan in s
print(np.isnan(u).any()) # False, no nan in u
print(np.isnan(v).any()) # False, no nan in v
