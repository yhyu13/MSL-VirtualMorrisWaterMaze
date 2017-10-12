from modelCA3C import *
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
    parser.add_argument('--num_workers', dest='num_workers',action='store',default=1,type=int)
    args = parser.parse_args()
    max_episode_length = 200
    batch_size = 1
    gamma = .99 # discount rate for advantage estimation and reward discounting
    s_size = 160*160
    a_size = 2 # Agent can turn or move with two floating number whose sign determines direction
    model_path = './model'
    gray = True
    load_model = args.load_model
    num_workers = args.num_workers
    noise =  OUNoise(2)# enable noisy dense layer to encourage exploration.
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

    with tf.device("/cpu:0"): 
        global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
        trainer_a = tf.train.AdamOptimizer(learning_rate=1e-3)
        trainer_c = tf.train.AdamOptimizer(learning_rate=5e-3)
        master_network_1 = Actor_Network(s_size,a_size,'global/actor',None,grayScale=gray) # Generate global network
        master_network_2 = Critic_Network(s_size,a_size,'global/critic',None,grayScale=gray)
        master_network_3 = Actor_Network(s_size,a_size,'global/actor/target',None,grayScale=gray)
        master_network_4 = Critic_Network(s_size,a_size,'global/critic/target',None,grayScale=gray)
        num_cpu = multiprocessing.cpu_count() # Set workers ot number of available CPU threads
        workers = []
        #replay_buffer = ReplayBuffer(1000)
            # Create worker classes
        for i in range(num_workers):
            # name,global_actor_target,global_critic_target,s_size,a_size,trainer_actor,trainer_critic,gamma,TAU,replay_buffer,model_path,global_episodes,noise,grayScale,is_training
            worker = Worker(i,master_network_3,master_network_4,s_size,a_size,trainer_a,trainer_c,0.99,1e-2,batch_size,None,model_path,global_episodes,noise=noise,grayScale=gray,is_training= True)
            workers.append(worker)
            worker.start(setting=0)
        saver = tf.train.Saver(max_to_keep=5)
        
    '''networks = ['global'] + ['worker_'+i for i in str(range(num_workers))]
    print(networks)'''
    #key = print_tensors_in_checkpoint_file('./tmp/checkpoints/mobilenet_v1_0.50_160.ckpt', tensor_name='',all_tensors=True)
    #print(key)
    
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        if load_model == True:
            print ('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(sess,ckpt.model_checkpoint_path)
        else:
            '''
            dict = {}
            value = slim.get_model_variables('global'+'/MobilenetV1')
            for variable in value:
                name = variable.name.replace('global'+'/','').split(':')[0]
                    #print(name)
                if name in key:
                    dict[name] = variable
                #print(dict)
                #print(dict)
            init_fn = slim.assign_from_checkpoint_fn(
                                os.path.join(checkpoints_dir, 'mobilenet_v1_0.50_160.ckpt'),
                                dict)
            init_fn(sess)'''
            sess.run(tf.global_variables_initializer())
            
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