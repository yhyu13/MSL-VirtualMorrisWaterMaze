
load_model = False # model's gonna use load_model as a global variable, so put it in front of model.
from model import *
import sys
import os
from inspect_checkpoints import print_tensors_in_checkpoint_file
checkpoints_dir = './tmp/checkpoints'

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

def main():
    if len(sys.argv) == 1:
        num_workers = 1
        noisy = True
    else:
        num_workers = int(sys.argv[1])
        noisy = str2bool(sys.argv[2])
    max_episode_length = 200
    gamma = .99 # discount rate for advantage estimation and reward discounting
    s_size = 160*160
    a_size = 3 # Agent can move Left, Right, or Straight
    model_path = './model'
    gray = True
    load_model = True
    print(" num_workers = %d" % num_workers)
    print(" noisy_enabled = %s" % str(noisy))
    
    print('''
    max_episode_length = 200
    gamma = .99 # discount rate for advantage estimation and reward discounting
    s_size = 160*160 
    a_size = 3 # Agent can move Left, Right, or Straight
    load_model = False
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
        trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
        master_network = AC_Network(s_size,a_size,'global',None,noisy,grayScale=gray) # Generate global network
        num_cpu = multiprocessing.cpu_count() # Set workers ot number of available CPU threads
        workers = []
            # Create worker classes
        for i in range(num_workers):
            worker = Worker(i,s_size,a_size,trainer,model_path,global_episodes,noisy,grayScale=gray,is_training= not load_model)
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