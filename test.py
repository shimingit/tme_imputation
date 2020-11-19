import tensorflow as tf

t1 = [[1, 2, 3], [4, 5, 6]]
t2 = [[7, 8, 9], [10, 11, 12]]

a = tf.concat([tf.expand_dims(t1,1), tf.expand_dims(t2,1)], 1)

sess = tf.Session()

print(a.shape)
print(sess.run(a))

b = tf.gather(a, [0,1])

print(sess.run(b))