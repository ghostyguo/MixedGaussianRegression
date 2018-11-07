import numpy as np
import matplotlib.pyplot as plt
import  tensorflow as tf
from numpy import genfromtxt


#init  
datafile = 'RMT_Mw_201805'

# Read CSV data
data = genfromtxt(datafile+'.csv', delimiter=',')
N=len(data)  
x_np = np.asarray(data[:,0]).reshape(N,1) #must be ordered to plot
y_np = np.asarray(data[:,1]) #must be ordered to plot
y_np=y_np/np.max(y_np)       #normalize to sum 1


# Generate tensorflow graph
tf.reset_default_graph()
with tf.name_scope("placeholders"):
  x = tf.placeholder(tf.float32, (N, 1))
  y = tf.placeholder(tf.float32, (N,))
with tf.name_scope("parameters"):  
  a1 = tf.Variable(1.0)  #tf.random_normal((1,)))  
  u1 = tf.Variable(2.5) #tf.random_normal((1,)))  
  s1 = tf.Variable(0.5) #tf.random_normal((1,)))
  a2 = tf.Variable(0.8)  #tf.random_normal((1,))) 
  u2 = tf.Variable(3.6) #tf.random_normal((1,))) 
  s2 = tf.Variable(0.5) #tf.random_normal((1,)))
with tf.name_scope("prediction"):
  y1_pred = tf.multiply(a1,tf.exp(tf.divide(tf.multiply(-1.0,tf.square(tf.subtract(x,u1))),tf.multiply(2.0,tf.multiply(s1,s1)))))
  y2_pred = tf.multiply(a2,tf.exp(tf.divide(tf.multiply(-1.0,tf.square(tf.subtract(x,u2))),tf.multiply(2.0,tf.multiply(s2,s2)))))
  y_pred = tf.add(y1_pred, y2_pred)
with tf.name_scope("loss"):
  l = tf.reduce_sum((y - tf.squeeze(y_pred))**2)
with tf.name_scope("optim"):
  #train_op = tf.train.AdamOptimizer(.001).minimize(l)  
  train_op = tf.train.GradientDescentOptimizer(.0001).minimize(l)
with tf.name_scope("summaries"):
  tf.summary.scalar("loss", l)
  merged = tf.summary.merge_all()
  
train_writer = tf.summary.FileWriter('d:/lr-train', tf.get_default_graph())

n_steps = 0
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  # Train model
  feed_dict = {x: x_np, y: y_np}
  min_loss=50000;
  update_loss=0
  while True:
    n_steps = n_steps + 1  
    update_loss =  update_loss + 1
    _, summary, loss = sess.run([train_op, merged, l], feed_dict=feed_dict)
    if (min_loss>loss):
        min_loss=loss
        update_loss=0
    if (n_steps%100==0):        
        a1_t, u1_t, s1_t, a2_t, u2_t, s2_t = sess.run([a1, u1, s1, a2, u2, s2])
        print("step %d, loss: %f, a1=%f u1=%f s1=%f a2=%f u2=%f s2=%f" % (n_steps, loss, a1_t, u1_t, s1_t, a2_t, u2_t, s2_t))
        train_writer.add_summary(summary, n_steps)
    if (loss<0.018 or update_loss>10000):
        break

  # Get weights
  a1_final, u1_final, s1_final, a2_final, u2_final, s2_final = sess.run([a1, u1, s1, a2, u2, s2])

  # Make Predictions
  y_pred_np = sess.run(y_pred, feed_dict={x: x_np})
  
y_pred_np = np.reshape(y_pred_np, -1)
print('n_steps=',n_steps)
print('a1_final=',a1_final,', u1_final=',u1_final,',s1_final=',s1_final)
print('a2_final=',a2_final,', u2_final=',u2_final,',s2_final=',s2_final)

# Now draw with learned regression curve
plt.clf()
plt.xlabel("x")
plt.ylabel("y")
plt.title("True Model versus Learned Model ")
plt.xlim(0, 10)
plt.scatter(x_np, y_np, color='b')
plt.plot(x_np,y_pred_np , color='r')
plt.savefig(datafile+'.png')