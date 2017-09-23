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
            
            hidden = slim.fully_connected(slim.flatten(self.conv3),128,activation_fn=tf.nn.relu)
            
            
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
			self.policy = slim.fully_connected(rnn_out,a_size,
				activation_fn=tf.nn.softmax,
				weights_initializer=normalized_columns_initializer(0.01),
				biases_initializer=None)
				
		self.q_gradient_input = tf.placeholder("float",[None,self.a_size])
		local_var = tf.get_collection(tf.Graphykeys.TRAINABLE_VARIABLES,self.scope)
		global_var = tf.get_collection(tf.Graphykeys.TRAINABLE_VARIABLES,'global/actor')
	    self.parameters_gradients,self.global_norm = tf.clip_by_global_norm(tf.gradients(self.policy,local_var,self.q_gradient_input),5.0)
		self.optimizer = trainer.apply_gradients(zip(self.parameters_gradienet,global_var))
		
	def pi(self,states,rnn_state):
		feed_dict = {self.inputs:states,self.state_in[0]:rnn_state[0],self.state_in[1]:rnn_state[1]}
		return sess.run(self.policy,feed_dict=feed_dict)
		
	def train(q_gradient_input,states,rnn_state):
		feed_dict = {self.q_gradient_input:q_gradient_input,self.inputs:states,self.state_in[0]:rnn_state[0],self.state_in[1]:rnn_state[1]}
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
            
			#Output layers value estimations
			self.value = slim.fully_connected(rnn_out,1,
				activation_fn=None,
				weights_initializer=normalized_columns_initializer(1.0),
				biases_initializer=None)
		
		# TRAINING
		self.y_input = tf.placeholder("float",[None,1])
		
		local_var = tf.get_collection(tf.Graphykeys.TRAINABLE_VARIABLES,self.scope)
		global_var = tf.get_collection(tf.Graphykeys.TRAINABLE_VARIABLES,'global/critic')
        weight_decay = tf.add_n([L2 * tf.nn.l2_loss(var) for var in local_var])
        self.cost = tf.reduce_mean(tf.square(self.y_input - self.value)) + weight_decay
		self.optimizer= trainer
        self.parameters_gradients,_ = zip(*self.optimizer.compute_gradients(self.cost,local_var))
        self.parameters_graidents,self.global_norm = tf.clip_by_global_norm(self.parameters_gradients,5.0)
        self.optimizer = self.optimizer.apply_gradients(zip(self.parameters_gradients,global_var))
        self.action_gradients = tf.gradients(self.value,self.inputs_action)
    
	def q_value(sess,states,actions,rnn_state):
		feed_dict = {self.inputs:states,self.inputs_action:actions,self.state_in[0]:rnn_state[0],self.state_in[1]:rnn_state[1]}
		return sess.run(self.value,feed_dict=feed_dict)
		
	def train(y_input,states,actions,rnn_state):
		feed_dict = {self.y_input:y_input,self.inputs:states,self.inputs_action:actions,self.state_in[0]:rnn_state[0],self.state_in[1]:rnn_state[1]}
		return sess.run([self.optimizer,self.cost],feed_dict=feed_dict)
		
	def gradient(state,actions,rnn_state):
		feed_dict = {self.inputs:states,self.inputs_action:actions,self.state_in[0]:rnn_state[0],self.state_in[1]:rnn_state[1]}
		return sess.run(self.action_gradients,feed_dict=feed_dict)
                
# VMWM Worker------------------------------------------------------------------------------------------------------------
class Worker():
    def __init__(self,name,global_actor_target,global_critic_target,s_size,a_size,trainer_actor,trainer_critic,TAU,replay_buffer,model_path,global_episodes,noise,grayScale,is_training):
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
        self.is_training = is_training
		self.replay_buffer = self.replay_buffer

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
        
    def train(self):
        episodes = np.array(self.memory_buffer.get_episodes())
		
        
		        # Sample a random minibatch of N sequences from replay buffer
        minibatch = self.replay_buffer.get_batch(BATCH_SIZE)
        # Construct histories
        observations = []
        next_observations = []
        actions = []
        rewards = []
        dones = []
        for each in minibatch:
            for i in range(1,len(each.observations)):
                observations.append(self.pad(each.observations[0:i]))
                next_observations.append(self.pad(each.observations[1,i+1]))
                actions.append(each.actions[0:i-1])
                rewards.append(each.rewards[0:i])
                if i == len(each.observations) - 1:
                    dones.append(True)
                else:
                    dones.append(False)
        # Calculate y_batch
        next_action_batch = self.actor_network.target_action(observations)
        q_value_batch = self.critic_network.target_q(next_observations,[self.pad(i+j) for (i,j) in zip(actions,next_action_batch)])
        y_batch = []
        for i in range(len(observations)):
            if dones[i]:
                y_batch.append(rewards[i][-1])
            else:
                y_batch.append(rewards[i][-1] + GAMMA * q_value_batch[i])
        y_batch = np.resize(y_batch,[len(observations),1])
        # Update critic by minimizing the loss L
        self.critic_network.train(y_batch,observations,[self.pad(i) for i in actions])

        # Update the actor policy using the sampled gradient:
        action_batch_for_gradients = self.actor_network.actions(observations)
        q_gradient_batch = self.critic_network.gradients(observations,action_batch_for_gradients)

        self.actor_network.train(q_gradient_batch,observations)

        # Update the target networks
        self.actor_network.update_target()
        self.critic_network.update_target()
		
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
                    '''
                    # If the episode hasn't ended, but the experience buffer is full, then we
                    # make an update step using that experience rollout.
                    if len(episode_buffer) == 30 and d != True: # change pisode length to 5, and try to modify Worker.train() function to utilize the next frame to train imagined frame.
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
                        '''
                    if d == True:
                        break
                        
                episode_len = self.env.get_episode_length()            
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_len)
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