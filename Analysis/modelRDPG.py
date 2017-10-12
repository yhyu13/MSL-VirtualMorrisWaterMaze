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

# Added by Andrew Liao
# for NoisyNet-DQN (using Factorised Gaussian noise)
# modified from ```dense``` function

# Actor Network------------------------------------------------------------------------------------------------------------
class Actor_Network():
    def __init__(self,s_size,a_size,scope,trainer,grayScale):
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
            self.conv1 = slim.conv2d(activation_fn=tf.nn.relu,
                inputs=self.imageIn,num_outputs=32,
                kernel_size=[5,5],stride=[2,2],padding='VALID')
            self.conv2 = slim.conv2d(activation_fn=tf.nn.relu,
                inputs=self.conv1,num_outputs=64,
                kernel_size=[5,5],stride=[2,2],padding='VALID')
            self.conv3 = slim.conv2d(activation_fn=tf.nn.relu,
                inputs=self.conv2,num_outputs=128,
                kernel_size=[5,5],stride=[2,2],padding='VALID')
            
            
            # change: Salient Object implemented, reference : https://arxiv.org/pdf/1704.07911.pdf , p3
            if scope == 'worker_0/actor':    
                
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
            
            hidden = slim.fully_connected(slim.flatten(self.conv3),128,activation_fn=tf.nn.relu)
            
            
            #Recurrent network for temporal dependencies
            lstm_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(128,dropout_keep_prob=1.0)
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
            self.policy = slim.fully_connected(rnn_out,a_size,
                activation_fn=tf.tanh,
                weights_initializer=normalized_columns_initializer(0.01),
                biases_initializer=None)
                
        self.q_gradient_input = tf.placeholder("float",[None,a_size])
        self.batch_size = tf.placeholder(shape=(),dtype=tf.int32)
        self.episode_len = tf.placeholder(shape=(),dtype=tf.int32)
        coff = tf.cast(tf.divide(1,tf.multiply(self.batch_size,self.episode_len)),tf.float32)
        local_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope)
        global_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'global/actor')
        self.parameters_gradients,self.global_norm = tf.clip_by_global_norm(tf.gradients(self.policy,local_var,self.q_gradient_input*coff),5.0)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).apply_gradients(zip(self.parameters_gradients,global_var))
        
    def pi(self,sess,states,rnn_state,salient_objects=False):
        feed_dict = {self.inputs:np.vstack(states),self.state_in[0]:rnn_state[0],self.state_in[1]:rnn_state[1]}
        if salient_objects:
            return sess.run([self.policy,self.state_out,self.salient_objects],feed_dict=feed_dict)
        else:
            return sess.run([self.policy,self.state_out],feed_dict=feed_dict)
        
    def train(self,sess,q_gradient_input,states,rnn_state,batch_size,episode_len):
        feed_dict = {self.q_gradient_input:q_gradient_input,self.inputs:np.vstack(states),self.state_in[0]:rnn_state[0],self.state_in[1]:rnn_state[1],self.batch_size:batch_size,self.episode_len:episode_len}
        return sess.run([self.optimizer],feed_dict=feed_dict)
        
# Actor Network------------------------------------------------------------------------------------------------------------
class Critic_Network():
    def __init__(self,s_size,a_size,scope,trainer,grayScale):
        with tf.variable_scope(scope):
            #Input and visual encoding layers
            self.inputs = tf.placeholder(shape=[None,s_size],dtype=tf.float32)
            self.inputs_action = tf.placeholder(shape=[None,a_size],dtype=tf.float32)
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
            self.conv1 = slim.conv2d(activation_fn=tf.nn.relu,
                inputs=self.imageIn,num_outputs=32,
                kernel_size=[5,5],stride=[2,2],padding='VALID')
            self.conv2 = slim.conv2d(activation_fn=tf.nn.relu,
                inputs=self.conv1,num_outputs=64,
                kernel_size=[5,5],stride=[2,2],padding='VALID')
            self.conv3 = slim.conv2d(activation_fn=tf.nn.relu,
                inputs=self.conv2,num_outputs=128,
                kernel_size=[5,5],stride=[2,2],padding='VALID')
            
            hidden = slim.fully_connected(slim.flatten(self.conv3),128,activation_fn=tf.nn.relu)
            hidden_action = slim.fully_connected(self.inputs_action,128,activation_fn=tf.nn.relu)
            hidden = tf.add(hidden,hidden_action)
            
            
            #Recurrent network for temporal dependencies
            lstm_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(128,dropout_keep_prob=1.0)
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
            
            #Output layers value estimations
            self.value = slim.fully_connected(rnn_out,1,
                activation_fn=None,
                weights_initializer=normalized_columns_initializer(1.0),
                biases_initializer=None)
        
        # TRAINING
        self.y_input = tf.placeholder("float",[None,1])
        self.batch_size = tf.placeholder(shape=(),dtype=tf.int32)
        self.episode_len = tf.placeholder(shape=(),dtype=tf.int32)
        coff = tf.cast(tf.divide(1,tf.multiply(self.batch_size,self.episode_len)),tf.float32)
        
        local_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope)
        global_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'global/critic')
        self.optimizer = tf.train.AdamOptimizer(learning_rate=5e-3)
        loss = coff*tf.reduce_sum(tf.multiply(tf.stop_gradient(self.y_input - self.value),self.value))
        self.parameters_gradients = tf.gradients(loss,local_var)
        self.parameters_graidents,self.global_norm = tf.clip_by_global_norm(self.parameters_gradients,5.0)
        self.optimizer = self.optimizer.apply_gradients(zip(self.parameters_gradients,global_var))
        self.action_gradients = tf.gradients(self.value,self.inputs_action)
    
    def q_value(self,sess,states,actions,rnn_state):
        feed_dict = {self.inputs:np.vstack(states),self.inputs_action:np.vstack(actions),self.state_in[0]:rnn_state[0],self.state_in[1]:rnn_state[1]}
        return sess.run([self.value,self.state_out],feed_dict=feed_dict)
        
    def train(self,sess,y_input,states,actions,rnn_state,batch_size,episode_len):
        feed_dict = {self.y_input:np.vstack(y_input),self.inputs:np.vstack(states),self.inputs_action:np.vstack(actions),self.state_in[0]:rnn_state[0],self.state_in[1]:rnn_state[1],self.batch_size:batch_size,self.episode_len:episode_len}
        return sess.run([self.optimizer],feed_dict=feed_dict)
        
    def action_gradient(self,sess,states,actions,rnn_state):
        feed_dict = {self.inputs:np.vstack(states),self.inputs_action:np.vstack(actions),self.state_in[0]:rnn_state[0],self.state_in[1]:rnn_state[1]}
        return sess.run(self.action_gradients,feed_dict=feed_dict)
                
# VMWM Worker------------------------------------------------------------------------------------------------------------
class Worker():
    def __init__(self,name,global_actor_target,global_critic_target,s_size,a_size,trainer_actor,trainer_critic,gamma,TAU,batch_size,replay_buffer,model_path,global_episodes,noise,grayScale,is_training):
        self.name = "worker_" + str(name)
        self.number = name        
        self.model_path = model_path
        
        self.trainer_actor = trainer_actor
        self.trainer_critic = trainer_critic
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter("train_"+str(self.number))
        self.noise = noise
        self.grayScale = grayScale
        self.gamma = gamma
        self.is_training = is_training
        self.batch_size = batch_size
        self.replay_buffer = replay_buffer

        #Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_Actor = Actor_Network(s_size,a_size,self.name+'/actor',trainer_actor,grayScale)
        self.local_Critic = Critic_Network(s_size,a_size,self.name+'/critic',trainer_critic,grayScale)
        # copy global networks parameters to local networks
        self.update_local_actor = update_target_graph('global/actor',self.name+'/actor')
        self.update_local_critic = update_target_graph('global/critic',self.name+'/critic')
        
        self.global_actor_target_network = global_actor_target
        self.global_critic_target_network = global_critic_target
        
        # update global target network
        self.update_target_network = update_target_network(TAU)        
        
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
        
    def train(self,sess):
        episodes = np.array(self.replay_buffer.get_batch(self.batch_size))
        
        # Construct histories
        # https://arxiv.org/pdf/1512.04455.pdf page 5.
        for ep in episodes:
            print(len(ep))
            observations = ep[:,0]
            actions = ep[:,1]
            rewards = ep[:,2]
            rnn_state = self.local_Actor.state_init
            y_batch = []
            for i in range(1,len(ep)):
                observations_t_1 = observations[0:i]
                # Calculate y_batch
                next_action,_ = self.global_actor_target_network.pi(sess,observations_t_1,rnn_state)
                q_value,_ = self.global_critic_target_network.q_value(sess,observations_t_1,next_action,rnn_state)
                y_batch.append(rewards[i-1] + self.gamma * q_value[-1,0])
            y_batch.append(rewards[-1])

            # Update critic by minimizing the loss L
            self.local_Critic.train(sess,y_batch,observations,actions,rnn_state,self.batch_size,len(ep))

            # Update the actor policy using the sampled gradient:
            action_batch_for_gradients,_ = self.local_Actor.pi(sess,observations,rnn_state)
            q_gradient_batch = self.local_Critic.action_gradient(sess,observations,action_batch_for_gradients,rnn_state)

            self.local_Actor.train(sess,q_gradient_batch[0],observations,rnn_state,self.batch_size,len(ep))

            # Update the target networks
            sess.run(self.update_target_network)
        
        
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
                
                sess.run(self.update_local_actor)
                sess.run(self.update_local_critic)
                episode_buffer = []
                episode_values = []
                episode_frames = []
                episode_reward = 0
                episode_len = 0
                d = False
                
                self.env.start_trial()
                sleep(0.1)
                
                s = self.env.get_screenImage()
                # change
                s1, s2 = None, s
                #episode_frames.append(s)
                s = process_frame(s)
                rnn_state_actor = self.local_Actor.state_init
                rnn_state_critic = self.local_Critic.state_init
                self.noise.reset()
                
                while self.env.is_episode_finished() == False:
                    #Take an action using probabilities from policy network output.
                    imageIn = np.expand_dims(s,axis=0)
                    if self.name == "worker_0":
                        a,rnn_state_actor,salient_objects = self.local_Actor.pi(sess,imageIn,rnn_state_actor,salient_objects=True)
                        v,rnn_state_critic = self.global_critic_target_network.q_value(sess,imageIn,a,rnn_state_critic)
                    else:
                        a,rnn_state_actor = self.local_Actor.pi(sess,imageIn,rnn_state,salient_objects=False)
                        v,rnn_state_critic = self.global_critic_target_network.q_value(sess,imageIn,a,rnn_state_critic)
                    #print(a_dist)
                    if self.is_training: a = np.clip(a[0] + self.noise.noise(),-1,1)
                    direcTurn,magniTurn,direcMove,speedMove = 0,abs(a[0]),0.,abs(a[1])
                    if magniTurn>0.1: # if magni is too small, then no turn
                        if a[0] > 0: 
                            direcTurn = 1
                        else:
                            direcTurn = -1
                    if speedMove>0.1: # if speed is too small, then no speed
                        direcMove = speedMove
                    
                    self.env.make_action(direcTurn,magniTurn*2,np.clip(direcMove*2,-1,1))
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
                        
                    episode_buffer.append([s,a,r])
                    episode_values.append(v[0,0])
                    self.env.display_value(v[0,0])

                    episode_reward += r
                    s = s1
                    total_steps += 1
     
                    if d == True:
                        break
                        
                if len(episode_buffer) > 2: # drop episdoes that are too short
                    self.replay_buffer.add(episode_buffer)
                episode_len = self.env.get_episode_length()            
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_len)
                self.episode_mean_values.append(np.mean(episode_values))
                
                # Update the network using the experience buffer at the end of the episode.
                if self.is_training and self.replay_buffer.count() > 1:
                    t1 = time()
                    for _ in range(1):
                        print("Training")
                        self.train(sess)
                    print(time()-t1)
                    
                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if episode_count % 5 == 0 and episode_count != 0:
                    mean_reward = np.mean(self.episode_rewards[-5:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    '''
                    if self.is_training:
                        summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                        summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                        summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                        summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                        summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                        '''
                        
                    self.summary_writer.add_summary(summary, episode_count)
                    self.summary_writer.flush()
                    
                    if self.name == 'worker_0' and (episode_count % 25 == 0 or not self.is_training):
                        time_per_step = 0.1 # Delay between action + 0.05 (unity delta time) * 2 (unity time scale)
                        images = np.array(episode_frames)
                        make_gif(images,'./frames/image'+str(episode_count)+'.gif',
                            duration=len(images)*time_per_step,true_image=True,salience=False)
                        sleep(1)
                        print("Episode "+str(episode_count)+" score: %d" % episode_reward)
                        print("Episodes so far mean reward: %d" % mean_reward)
                    if episode_count % 100 == 0 and self.name == 'worker_0' and self.is_training:
                        saver.save(sess,self.model_path+'/model-'+str(episode_count)+'.cptk')
                        print ("Saved Model")
                        sleep(1)
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