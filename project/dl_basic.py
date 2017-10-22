import tensorflow as tf

x1 = tf.constant(5)
x2 = tf.constant(6)

##result = x1*x2

#COMPUTATION GRAPH

result = tf.multiply(x1,x2)
print(result)   #abstract tensor

##sess = tf.Session()   ##creates a session variable
##print(sess.run(result))     #runs session to display output
##sess.close()   ## like a file after running a session it should be closed

#WHAT WILL HAPPEN IN THE SESSION
with tf.Session() as sess:
    print(sess.run(result))
