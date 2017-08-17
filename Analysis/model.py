import threading
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal
import socket

from helper import *
from envVMWM import *
#import mobileNet
exe_location='C:\\Users\\YuHang\\Desktop\\Water_Maze\\v0.18\\VMWM.exe'
cfg_location = 'C:\\Users\\YuHang\\Desktop\\Water_Maze\\v0.18\\VMWM_data\\configuration_original.txt'

from random import choice
from time import sleep
from time import time
import cv2
 
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

# ================================================================
# Model components
# ================================================================

# Added by Andrew Liao
# for NoisyNet-DQN (using Factorised Gaussian noise)
# modified from ```dense``` function
def sample_noise(shape):
    noise = np.random.normal(size=shape).astype(np.float32)
    #noise = np.ones(size=shape).astype(np.float32) # whenever not in training, simply return a matrix of ones. 
    return noise
    
global_p_a = 0.
global_q_a = 0.
global_p_v = 0.
global_q_v = 0.
    
def noisy_dense(x, size, name, bias=True, activation_fn=tf.identity, factorized=False):

    global global_p_a
    global global_q_a
    global global_p_v
    global global_q_v
    # https://arxiv.org/pdf/1706.10295.pdf page 4
    # the function used in eq.7,8 : f(x)=sgn(x)*sqrt(|x|)
    def f(x):
        return tf.multiply(tf.sign(x), tf.pow(tf.abs(x), 0.5))
    # Initializer of \mu and \sigma 
    mu_init = tf.random_uniform_initializer(minval=-1*1/np.power(x.get_shape().as_list()[1], 0.5),     
                                                maxval=1*1/np.power(x.get_shape().as_list()[1], 0.5))
    sigma_init = tf.constant_initializer(0.4/np.power(x.get_shape().as_list()[1], 0.5))
    # Sample noise from gaussian
    if name == 'global':
        if size == 3: # check condition
            p = sample_noise([128, 3]) # 256 is rnn_size
            global_p_a = p
            q = sample_noise([1, 3]) # 3 is action size
            global_q_a = q
        else:
            p = sample_noise([128, 1]) # 256 is rnn_size
            global_p_v = p
            q = sample_noise([1, 1]) # 1 is value size
            global_q_v = q
    else: # for actors, copy p & q from the global network
        if size == 3: # check condition
            p = global_p_a
            q = global_q_a
        else:
            p = global_p_v
            q = global_q_v
    
    f_p = f(p); f_q = f(q)
    w_epsilon = f_p*f_q; b_epsilon = tf.squeeze(f_q)
    if not factorized: # just resample the noisy matrix to get independent guassian noise
        w_epsilon = tf.identity(sample_noise(w_epsilon.get_shape().as_list()))
    # w = w_mu + w_sigma*w_epsilon
    options = {3:'action',1:'value'}
    w_mu = tf.get_variable(name + "/w_mu" + options[size], [x.get_shape()[1], size], initializer=mu_init)
    w_sigma = tf.get_variable(name + "/w_sigma" + options[size], [x.get_shape()[1], size], initializer=sigma_init)
    w = w_mu + tf.multiply(w_sigma, w_epsilon)
    ret = tf.matmul(x, w)
    if bias:
        # b = b_mu + b_sigma*b_epsilon
        b_mu = tf.get_variable(name + "/b_mu" + options[size], [size], initializer=mu_init)
        b_sigma = tf.get_variable(name + "/b_sigma" + options[size], [size], initializer=sigma_init)
        b = b_mu + tf.multiply(b_sigma, b_epsilon)
        return activation_fn(ret + b)
    else:
        return activation_fn(ret)
    
def use_mobileNet(inputs):
        logits, salient_objects = mobileNet.mobilenet_v1_050(inputs,
                                             is_training=False,
                                             prediction_fn=None)
        return logits, salient_objects
# Actor Network------------------------------------------------------------------------------------------------------------
class AC_Network():
    def __init__(self,s_size,a_size,scope,trainer,noisy,grayScale):
        with tf.variable_scope(scope):
            #Input and visual encoding layers
            self.inputs = tf.placeholder(shape=[None,s_size],dtype=tf.float32)
            if grayScale:
                self.imageIn = tf.reshape(self.inputs,shape=[-1,160,160,1])
            else:
                self.imageIn = tf.reshape(self.inputs,shape=[-1,160,160,3])
            
            # Create the model, use the default arg scope to configure the batch norm parameters.
            '''
            logits,self.salient_objects = use_mobileNet(self.imageIn)
            self.logits = logits[0]
            
            hidden = tf.nn.tanh(logits)
            '''
            self.conv1 = slim.conv2d(activation_fn=tf.nn.elu,
                inputs=self.imageIn,num_outputs=32,
                kernel_size=[5,5],stride=[2,2],padding='VALID')
            self.conv2 = slim.conv2d(activation_fn=tf.nn.elu,
                inputs=self.conv1,num_outputs=64,
                kernel_size=[5,5],stride=[2,2],padding='VALID')
            self.conv3 = slim.conv2d(activation_fn=tf.nn.elu,
                inputs=self.conv2,num_outputs=128,
                kernel_size=[5,5],stride=[2,2],padding='VALID')
            
            
            # change: Salient Object implemented, reference : https://arxiv.org/pdf/1704.07911.pdf , p3
            if scope == 'worker_0':    
                
                feature_maps_avg3 = tf.reduce_mean(tf.nn.relu(self.conv3), axis=(0,3),keep_dims=True)
                #print(feature_maps_avg3.get_shape())
                feature_maps_avg2 = tf.reduce_mean(tf.nn.relu(self.conv2), axis=(0,3),keep_dims=True)
                #print(feature_maps_avg2.get_shape())
                feature_maps_avg1 = tf.reduce_mean(tf.nn.relu(self.conv1), axis=(0,3),keep_dims=True)
                #print(feature_maps_avg1.get_shape())

                scale_up_deconv3 = tf.stop_gradient(tf.nn.conv2d_transpose(feature_maps_avg3,np.ones([5,5,1,1]).astype(np.float32), output_shape=feature_maps_avg2.get_shape().as_list(), strides=[1,2,2,1],padding='VALID'))
                #print(scale_up_deconv3)
                scale_up_deconv2 = tf.stop_gradient(tf.nn.conv2d_transpose(tf.multiply(feature_maps_avg2,scale_up_deconv3),np.ones([5,5,1,1]).astype(np.float32), output_shape=feature_maps_avg1.get_shape().as_list(), strides=[1,2,2,1],padding='VALID'))
                #print(scale_up_deconv2)
                self.salient_objects = tf.stop_gradient(tf.squeeze(tf.nn.conv2d_transpose(tf.multiply(feature_maps_avg1,scale_up_deconv2),np.ones([5,5,1,1]).astype(np.float32), output_shape=[1,160,160,1],strides=[1,2,2,1],padding='VALID')))
            '''
            # Augst 1st
            # 1, https://arxiv.org/pdf/1611.07078.pdf A DEEP LEARNING APPROACH FOR JOINT VIDEO FRAME AND REWARD PREDICTION IN ATARI GAMES
            # 2, https://arxiv.org/pdf/1707.06203.pdf The I2A architecture
            # Now, TO DO: Implement an additional NN that can imagine 2 steps ahead'''
            
            # All convNet hidden state
            '''
            kernel_size = mobileNet._reduced_kernel_size_for_small_input(self.conv3, [7 ,7])
            net = slim.avg_pool2d(self.conv3, kernel_size, padding='VALID',
                                  scope='AvgPool_1a')
            net = slim.dropout(net, keep_prob=0.8, scope='Dropout_1b')
            logits = slim.conv2d(net, 128, [1,1], activation_fn=None,
                                 normalizer_fn=None, scope='Conv2d_1c_1x1')
            logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
            hidden = tf.nn.tanh(logits)'''
            
            hidden = slim.fully_connected(slim.flatten(self.conv3),128,activation_fn=tf.nn.elu)
            
            
            #Recurrent network for temporal dependencies
            lstm_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(128,dropout_keep_prob=0.8)
            #lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell,output_keep_prob=0.5)
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = (c_in, h_in)
            rnn_in = tf.expand_dims(hidden, [0])
            step_size = tf.shape(self.imageIn)[:1]
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,
                time_major=False)
            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            rnn_out = tf.reshape(lstm_outputs, [-1, 128])
            
            # try the case without lstm
            #rnn_out = hidden
            
            if noisy:
                # Apply noisy network on fully connected layers
                # ref: https://arxiv.org/abs/1706.10295
                self.policy = noisy_dense(rnn_out,name=scope, size=a_size, activation_fn=tf.nn.softmax)
                self.value = noisy_dense(rnn_out,name=scope, size=1) # default activation_fn=tf.identity
            else:
                #Output layers for policy and value estimations
                self.policy = slim.fully_connected(rnn_out,a_size,
                    activation_fn=tf.nn.softmax,
                    weights_initializer=normalized_columns_initializer(0.8),
                    biases_initializer=None)
                self.value = slim.fully_connected(rnn_out,1,
                    activation_fn=None,
                    weights_initializer=normalized_columns_initializer(1.0),
                    biases_initializer=None)
                    
            #Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions,a_size,dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None],dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None],dtype=tf.float32)

                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

                #Loss functions
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value,[-1])))
                self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs)*self.advantages)
                if noisy:
                    self.entropy = tf.constant(42.,dtype=tf.float32)
                    self.loss = 0.5 * self.value_loss + self.policy_loss
                else:
                    self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy))
                    self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

                #Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss,local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,40.0)
                
                #Apply local gradients to global network
                #Comment these two lines out to stop training
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads,global_vars))
                
# VMWM Worker------------------------------------------------------------------------------------------------------------
class Worker():
    def __init__(self,name,s_size,a_size,trainer,model_path,global_episodes,noisy,grayScale,is_training):
        self.name = "worker_" + str(name)
        self.number = name        
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter("train_"+str(self.number))
        self.noisy = noisy
        self.grayScale = grayScale
        self.is_training = is_training

        #Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = AC_Network(s_size,a_size,self.name,trainer,noisy,grayScale)
        self.update_local_ops = update_target_graph('global',self.name)        
        
        #Set up actions
        self.actions = np.identity(a_size,dtype=bool).tolist()
        
        #Set up VMWM env
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
        #print(sock.getsockname())
        #sock.shutdown(socket.SHUT_WR)
        self.env = VMWMGame(cfg_location,exe_location)
        self.env.reset_cfg()
        self.env.set_trial('Practice - Hills')
        self.env.set_local_host('127.0.0.1', port) # local host IP address & dynamic allocated port 
        
    def start(self,setting=0):
        self.env.start(self.grayScale)
        if self.name == "worker_0":
            # Set up OpenCV Window
            cv2.startWindowThread()
        
    def train(self,rollout,sess,gamma,bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:,0]
        actions = rollout[:,1]
        rewards = rollout[:,2]
        next_observations = rollout[:,3] # Aug 1st, notice next observation is never used
        values = rollout[:,5]
        
        # Here we take the rewards and values from the rollout, and use them to 
        # generate the advantage and discounted returns. 
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus,gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages,gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        rnn_state = self.local_AC.state_init
        feed_dict = {self.local_AC.target_v:discounted_rewards,
            self.local_AC.inputs:np.vstack(observations),
            self.local_AC.actions:actions,
            self.local_AC.advantages:advantages,
            self.local_AC.state_in[0]:rnn_state[0],
            self.local_AC.state_in[1]:rnn_state[1]}
        v_l,p_l,e_l,g_n,v_n,_ = sess.run([self.local_AC.value_loss,
            self.local_AC.policy_loss,
            self.local_AC.entropy,
            self.local_AC.grad_norms,
            self.local_AC.var_norms,
            self.local_AC.apply_grads],
            feed_dict=feed_dict)
        return v_l / len(rollout),p_l / len(rollout),e_l / len(rollout), g_n,v_n
        
    def work(self,max_episode_length,gamma,sess,coord,saver):
        if self.is_training:
            episode_count = sess.run(self.global_episodes)
        else:
            episode_count = 0
        wining_episode_count = 0
        total_steps = 0
        print ("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():
            #not_start_training_yet = True
            while not coord.should_stop():
                
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_frames = []
                episode_reward = 0
                episode_step_count = 0
                d = False
                
                self.env.start_trial()
                sleep(0.1)
                
                s = self.env.get_screenImage()
                # change
                s1, s2 = None, s
                #episode_frames.append(s)
                s = process_frame(s)
                rnn_state = self.local_AC.state_init
                
                while self.env.is_episode_finished() == False:
                    #Take an action using probabilities from policy network output.
                    ''' # for MobileNet only
                    if self.name == "worker_0":
                        a_dist,v,rnn_state,logits,salient_objects = sess.run([self.local_AC.policy,self.local_AC.value,self.local_AC.state_out,self.local_AC.logits,self.local_AC.salient_objects], 
                            feed_dict={self.local_AC.inputs:[s],
                            self.local_AC.state_in[0]:rnn_state[0],
                            self.local_AC.state_in[1]:rnn_state[1]})
                    else:
                        a_dist,v,rnn_state,logits = sess.run([self.local_AC.policy,self.local_AC.value,self.local_AC.state_out,self.local_AC.logits], 
                            feed_dict={self.local_AC.inputs:[s],
                            self.local_AC.state_in[0]:rnn_state[0],
                            self.local_AC.state_in[1]:rnn_state[1]})'''
                            
                    if self.name == "worker_0":
                        a_dist,v,rnn_state,salient_objects = sess.run([self.local_AC.policy,self.local_AC.value,self.local_AC.state_out,self.local_AC.salient_objects], 
                            feed_dict={self.local_AC.inputs:[s],
                            self.local_AC.state_in[0]:rnn_state[0],
                            self.local_AC.state_in[1]:rnn_state[1]})
                    else:
                        a_dist,v,rnn_state = sess.run([self.local_AC.policy,self.local_AC.value,self.local_AC.state_out], 
                            feed_dict={self.local_AC.inputs:[s],
                            self.local_AC.state_in[0]:rnn_state[0],
                            self.local_AC.state_in[1]:rnn_state[1]})
                    #print(a_dist)
                    a = np.random.choice(a_dist[0],p=a_dist[0])
                    a = np.argmax(a_dist == a)
                    '''
                    probs = softmax(logits)
                    index = np.argmax(probs)
                    print("Object: {}. Confidence: {}. Mean: {}. Std: {}.".format(label_dict[index], probs[index], np.mean(probs),np.std(probs)))
                    '''
                    self.env.make_action(a,150)
                    r = self.env.get_reward()
                    # change
                    #sleep(0.05)
                    d = self.env.is_episode_finished()
                    if d == False:
                        s1 = self.env.get_screenImage()
                        # change 
                        if self.name == "worker_0":
                            #print(np.ndim(salient_objects)) == 2
                            s2 = mask_color_img(s2,process_salient_object(np.asarray(salient_objects)),self.grayScale)
                            cv2.imshow('frame', s2)
                            cv2.waitKey(1)
                            episode_frames.append(s2)
                        #else:
                            #episode_frames.append(s1)
                            
                        s2 = s1
                        s1 = process_frame(s1)
                    else:
                        s1 = s
                        
                    episode_buffer.append([s,a,r,s1,d,v[0,0]])
                    episode_values.append(v[0,0])
                    self.env.display_value(v[0,0])

                    episode_reward += r
                    s = s1
                    total_steps += 1
                    episode_step_count += 1
                    
                    # If the episode hasn't ended, but the experience buffer is full, then we
                    # make an update step using that experience rollout.
                    if len(episode_buffer) == 30 and d != True and episode_step_count != max_episode_length - 1: # change pisode length to 5, and try to modify Worker.train() function to utilize the next frame to train imagined frame.
                        # Since we don't know what the true final return is, we "bootstrap" from our current
                        # value estimation.
                        v1 = sess.run(self.local_AC.value, 
                            feed_dict={self.local_AC.inputs:[s],
                            self.local_AC.state_in[0]:rnn_state[0],
                            self.local_AC.state_in[1]:rnn_state[1]})[0,0]
                        if self.is_training:
                            v_l,p_l,e_l,g_n,v_n = self.train(episode_buffer,sess,gamma,v1)
                        episode_buffer = []
                        sess.run(self.update_local_ops)
                        
                    if d == True:
                        break
                                   
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))
                
                # Update the network using the experience buffer at the end of the episode.
                if len(episode_buffer) != 0:
                    if self.is_training:
                        v_l,p_l,e_l,g_n,v_n = self.train(episode_buffer,sess,gamma,0.0)
                                
                    
                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if episode_count % 5 == 0 and episode_count != 0:
                    mean_reward = np.mean(self.episode_rewards[-5:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    if self.is_training:
                        summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                        summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                        summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                        summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                        summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                    self.summary_writer.add_summary(summary, episode_count)
                    self.summary_writer.flush()
                    
                    if self.name == 'worker_0' and (episode_count % 25 == 0 or not self.is_training):
                        time_per_step = 0.1 # Delay between action + 0.05 (unity delta time) * 2 (unity time scale)
                        images = np.array(episode_frames)
                        make_gif(images,'./frames/image'+str(episode_count)+'.gif',
                            duration=len(images)*time_per_step,true_image=True,salience=False)
                        sleep(5)
                        print("Episode "+str(episode_count)+" score: %d" % episode_reward)
                        print("Episodes so far mean reward: %d" % mean_reward)
                    if episode_count % 25 == 0 and self.name == 'worker_0' and self.is_training:
                        saver.save(sess,self.model_path+'/model-'+str(episode_count)+'.cptk')
                        print ("Saved Model")
                        sleep(15)
                if self.name == 'worker_0' and self.is_training:
                    sess.run(self.increment)
                    
                episode_count += 1
                
                if self.name == "worker_0" and episode_reward > 100. and not self.is_training:
                    wining_episode_count += 1
                    print('Worker_0 find the platform in Episode {}! Total percentage of finding the platform is: {}%'.format(episode_count, int(wining_episode_count / episode_count * 100)))
                    
                
                #not_start_training_yet = False # Yes, we did training the first time, now we can broadcast cv2
                # Start a new episode
                self.env.new_episode()
            
            # All done Stop trail
            self.env.end_trial()
            self.env.s.close()
            # change
            if self.name == "worker_0":
                cv2.destroyAllWindows()
            # Confirm exit
            print('Done '+self.name)