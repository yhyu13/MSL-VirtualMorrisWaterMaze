# Thanks to Arthur Juliani for offering greate introduction to A3C architecture:
# https://github.com/awjuliani/DeepRL-Agents/blob/master/A3C-Doom.ipynb

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
from replay_buffer import *
from ou_noise import *
#import mobileNet
exe_location='C:\\Users\\YuHang\\Desktop\\Water_Maze\\v0.31\\VMWM.exe'
cfg_location = 'C:\\Users\\YuHang\\Desktop\\Water_Maze\\v0.31\\VMWM_data\\configuration_original.txt'

from random import choice
from time import sleep
from time import time
import cv2


# ================================================================
# Model components
# ================================================================

# Actor Network------------------------------------------------------------------------------------------------------------
class AC_Network():
    def __init__(self,s_size,a_size,scope,trainer,grayScale=True):
        self.trainer = trainer
        
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
            #rnn_out = tf.tanh(hidden)
            
            #Output layers for policy estimations
            self.mu = slim.fully_connected(rnn_out,a_size,
                activation_fn=None,
                weights_initializer=normalized_columns_initializer(0.01),
                biases_initializer=None)
            
            self.var = slim.fully_connected(rnn_out,a_size,
                activation_fn=tf.nn.softplus,
                weights_initializer=normalized_columns_initializer(1.),
                biases_initializer=None)
                
            self.normal_dist = tf.contrib.distributions.Normal(self.mu, tf.sqrt(self.var))
            self.policy = self.normal_dist.sample(1)
            self.value = slim.fully_connected(rnn_out,1,
                activation_fn=None,
                weights_initializer=normalized_columns_initializer(1.),
                biases_initializer=None)
                
                    #Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.actions = tf.placeholder(shape=[None,a_size],dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None],dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None],dtype=tf.float32)

                #Loss functions
                self.value_loss = tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value,[-1])))
                self.log_prob = tf.reduce_sum(self.normal_dist.log_prob(self.actions))
                self.entropy = tf.multiply(0.5,tf.add(tf.log(6.28*self.var),1))  # encourage exploration

                self.policy_loss = tf.add(-tf.reduce_sum(self.log_prob*self.advantages),0.01 * self.entropy)

                self.loss = tf.add(self.value_loss,self.policy_loss) 

                #Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss,local_vars)
                
                #self.var_norms = tf.global_norm(local_vars)
                self.gradients,self.grad_norms = tf.clip_by_global_norm(self.gradients,40)
                
                #Apply local gradients to global network
                #Comment these two lines out to stop training
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = self.trainer.apply_gradients(zip(self.gradients,global_vars))
                
# VMWM Worker------------------------------------------------------------------------------------------------------------
class Worker():
    def __init__(self,name,s_size,a_size,trainer,gamma,TAU,batch_size,replay_buffer,model_path,global_episodes,grayScale,is_training):
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
        self.grayScale = grayScale
        self.gamma = gamma
        self.is_training = is_training
        self.batch_size = batch_size
        self.replay_buffer = replay_buffer

        #Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = AC_Network(s_size,a_size,self.name,trainer,self.grayScale)
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
        self.env.set_trial('Test Trial')
        self.env.set_local_host('127.0.0.1', port) # local host IP address & dynamic allocated port 
        
    def start(self,setting=0):
        self.env.start(self.grayScale)
        #if self.name == "worker_0":
            # Set up OpenCV Window
            #cv2.startWindowThread()
        
    def train(self,rollout,sess,gamma,bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:,0]
        actions = rollout[:,1]
        rewards = rollout[:,2]
        # reward clipping:  scale and clip the values of the rewards to the range -1,+1
        #rewards = (rewards - np.mean(rewards)) / np.max(abs(rewards))

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
        rnn_state = self.local_AC.state_init
        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        feed_dict = {self.local_AC.target_v:discounted_rewards,
            self.local_AC.inputs:np.vstack(observations),
            self.local_AC.actions:np.vstack(actions),
            self.local_AC.advantages:advantages,
            self.local_AC.state_in[0]:rnn_state[0],
            self.local_AC.state_in[1]:rnn_state[1]}
        mu,var,v_l,p_l,e_l,g_n,_ = sess.run([self.local_AC.mu,self.local_AC.var,self.local_AC.value_loss,
            self.local_AC.policy_loss,
            self.local_AC.entropy,
            self.local_AC.grad_norms,
            self.local_AC.apply_grads],
            feed_dict=feed_dict)
        return np.mean(mu),np.mean(var),np.mean(v_l / len(rollout)),np.mean(p_l / len(rollout)),np.mean(e_l / len(rollout)), g_n
        
        
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
            
                if episode_count >= 5001:
                    break
                
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_frames = []
                episode_reward = 0
                episode_len = 0
                d = False
                
                self.env.start_trial()
                sleep(0.01)
                
                s = self.env.get_screenImage()
                # change
                s1, s2 = None, s
                #episode_frames.append(s)
                s = process_frame(s)
                rnn_state = self.local_AC.state_init
                
                while self.env.is_episode_finished() == False:
                    #Take an action using probabilities from policy network output.
                    if self.name == "worker_0":
                        a,v,rnn_state,salient_objects = sess.run([self.local_AC.policy,self.local_AC.value,self.local_AC.state_out,self.local_AC.salient_objects], 
                            feed_dict={self.local_AC.inputs:[s],
                            self.local_AC.state_in[0]:rnn_state[0],
                            self.local_AC.state_in[1]:rnn_state[1]})
                    else:
                        a,v,rnn_state = sess.run([self.local_AC.policy,self.local_AC.value,self.local_AC.state_out], 
                            feed_dict={self.local_AC.inputs:[s],
                            self.local_AC.state_in[0]:rnn_state[0],
                            self.local_AC.state_in[1]:rnn_state[1]})
                    #print(a_dist)
                    if self.is_training: a = a[0,0]
                    direcTurn,magniTurn = 0,abs(a[0])
                    #print(a)
                    if magniTurn>0.1: # if magni is too small, then no turn
                        if a[0] > 0: 
                            direcTurn = 2
                        else:
                            direcTurn = 0
                    a = [np.round(np.minimum(magniTurn*10,100),decimals=2),np.clip(a[1],0.1,1)]
                    self.env.make_action(direcTurn,a[0],a[1])
                    sleep(0.01)
                    r = self.env.get_reward()
                    if r is None: r = 0.0
                    sleep(0.01)
                    d = self.env.is_episode_finished()
                    if d == False:
                        #print("here")
                        s1 = self.env.get_screenImage()
                        # change 
                        if self.name == "worker_0":
                            #print(np.ndim(salient_objects)) == 2
                            s2 = mask_color_img(s2,process_salient_object(np.asarray(salient_objects)),self.grayScale)
                            #cv2.imshow('frame', s2)
                            #cv2.waitKey(1)
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
                    
                    if len(episode_buffer) == 15 and d != True: # change pisode length to 5, and try to modify Worker.train() function to utilize the next frame to train imagined frame.
                        # Since we don't know what the true final return is, we "bootstrap" from our current
                        # value estimation.
                        v1 = sess.run(self.local_AC.value, 
                            feed_dict={self.local_AC.inputs:[s],
                            self.local_AC.state_in[0]:rnn_state[0],
                            self.local_AC.state_in[1]:rnn_state[1]})[0,0]
                        if self.is_training:
                            mu,var,v_l,p_l,e_l,g_n = self.train(episode_buffer,sess,gamma,v1)
                        episode_buffer = []
                        sess.run(self.update_local_ops)
     
                    if d == True:
                        break
                
                episode_len = self.env.get_episode_length()
                # if not receive length, then length equals max length
                if episode_len is None: episode_len = 7.5
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_len)
                self.episode_mean_values.append(np.mean(episode_values))
                
                # Update the network using the experience buffer at the end of the episode.
                if len(episode_buffer) != 0:
                    if self.is_training:
                        mu,var,v_l,p_l,e_l,g_n = self.train(episode_buffer,sess,gamma,0.0)
                        #print(l,v_l,p_l,e_l,g_n)
                    
                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if episode_count % 5 == 0 and episode_count != 2000:
                    mean_reward = np.mean(self.episode_rewards[-5:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    
                    if self.is_training:
                        summary.value.add(tag='Policy/mu', simple_value=float(mu))
                        summary.value.add(tag='Policy/var', simple_value=float(var))
                        summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                        summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                        summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                        summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                    
                        
                    self.summary_writer.add_summary(summary, episode_count)
                    self.summary_writer.flush()
                    
                    if self.name == 'worker_0' and (episode_count % 100 == 0 or not self.is_training):
                        time_per_step = 0.1 # Delay between action + 0.05 (unity delta time) * 2 (unity time scale)
                        images = np.array(episode_frames)
                        make_gif(images,'./frames/image'+str(episode_count)+'.gif',
                            duration=len(images)*time_per_step,true_image=True,salience=False)
                        #sleep(0.1)
                        print("Episode "+str(episode_count)+" score: %d" % episode_reward)
                        print("Episodes so far mean reward: %d" % mean_reward)
                    if episode_count % 200 == 0 and self.name == 'worker_0' and self.is_training:
                        saver.save(sess,self.model_path+'/model-'+str(episode_count)+'.cptk')
                        print ("Saved Model")
                        #sleep(0.1)
                if self.name == 'worker_0' and self.is_training:
                    sess.run(self.increment)
                
                episode_count += 1

                #not_start_training_yet = False # Yes, we did training the first time, now we can broadcast cv2
                # Start a new episode
                self.env.new_episode()
            
            # All done Stop trail
            self.env.end_trial()
            self.env.s.close()
            # change
            #if self.name == "worker_0":
                #cv2.destroyAllWindows()
            # Confirm exit
            print('Done '+self.name)