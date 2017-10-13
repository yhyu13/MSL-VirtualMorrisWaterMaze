from modelA3C_2 import *
import sys
import os
#from inspect_checkpoints import print_tensors_in_checkpoint_file
checkpoints_dir = './tmp/checkpoints'
import argparse
from replay_buffer import *
from ou_noise import *

def main():

    parser = argparse.ArgumentParser(description='Train or test neural net motor controller')
    parser.add_argument('--load_model', dest='load_model', action='store_true', default=False)
    parser.add_argument('--num_workers', dest='num_workers',action='store',default=4,type=int)
    args = parser.parse_args()
    max_episode_length = 200
    batch_size = 1
    gamma = .995 # discount rate for advantage estimation and reward discounting
    s_size = 160*160
    a_size = 3 # Agent can turn or move with two floating number whose sign determines direction
    model_path = './model'
    gray = True
    load_model = args.load_model
    num_workers = args.num_workers
    print(" num_workers = %d" % num_workers)
    
    print('''
    gamma = .99 # discount rate for advantage estimation and reward discounting
    s_size = 160*160 
    a_size = 2 # Agent can move Left, Right, or Straight
    model_path = './model'
    ''')

    tf.reset_default_graph()

    if not os.path.exists(model_path):
        os.makedirs(model_path)
        
    #Create a directory to save episode playback gifs to
    
    if not os.path.exists('./frames'):
        os.makedirs('./frames')
        
    '''networks = ['global'] + ['worker_'+i for i in str(range(num_workers))]
    print(networks)'''
    #key = print_tensors_in_checkpoint_file('./tmp/checkpoints/mobilenet_v1_0.50_160.ckpt', tensor_name='',all_tensors=True)
    #print(key)
    
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        
        with tf.device("/cpu:0"): 
            global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
            trainer = tf.train.AdamOptimizer(learning_rate=1e-3)
            master_network = AC_Network(s_size,a_size,'global',None,grayScale=gray) # Generate global network
            num_cpu = multiprocessing.cpu_count() # Set workers ot number of available CPU threads
            workers = []
            #replay_buffer = ReplayBuffer(1000)
                # Create worker classes
            for i in range(num_workers):
                # name,global_actor_target,global_critic_target,s_size,a_size,trainer_actor,trainer_critic,gamma,TAU,replay_buffer,model_path,global_episodes,noise,grayScale,is_training
                worker = Worker(i,s_size,a_size,trainer,gamma,1e-2,batch_size,None,model_path,global_episodes,grayScale=gray,is_training= True)
                workers.append(worker)
                worker.start(setting=0)
            saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')+[global_episodes],max_to_keep=0)
            sess.run(tf.global_variables_initializer())
            if load_model == True:
                print ('Loading Model...')
                ckpt = tf.train.get_checkpoint_state(model_path)
                saver.restore(sess,ckpt.model_checkpoint_path)
            
            
        # This is where the asynchronous magic happens.
        # Start the "work" process for each worker in a separate thread.
        worker_threads = []
        for worker in workers:
            worker_work = lambda: worker.work(max_episode_length,gamma,sess,coord,saver)
            t = threading.Thread(target=(worker_work))
            t.start()
            sleep(0.5)
            worker_threads.append(t)
        coord.join(worker_threads)
        
if __name__ == "__main__":
    main()