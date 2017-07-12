'''
    Created May 30th, 2017 Hang (Yohan) Yu | Physics @ UIUC
    
    The mitivation of encapsulating the training environment is from the spirit of control theory,
    
'''
import io
import os
import socket
import select
import cv2
import time
import matplotlib.pyplot as plt
import threading
import numpy as np
from subprocess import Popen

class VMWMGame:
    def __init__(self, config_original_txt_location, VMWMExe_location):
        '''
            initialization & set the path of the original configuraiton text
            set unity program location (.exe)
        '''
        self.cfg_txt = config_original_txt_location
        self.exe_location = VMWMExe_location
        
    def reset_cfg(self):
        '''
            open the original configuration file (.txt version) & modify it into .ini extension so to enable the environment
        '''
        
        thisFile1 = self.cfg_txt
        thisFile2 = "\\".join(self.cfg_txt.split('\\')[:-1]) + "\\configuration.txt"
        with open(thisFile1) as f:
            with open(thisFile2, "w") as f1:
                for line in f:
                    f1.write(line)
        
        base = os.path.splitext(thisFile2)[0]
        # configuration.ini already existed, remove it in order to reset successfully
        if os.path.isfile(base + ".ini"):
            os.remove(base + ".ini")
        # rename .txt to .ini
        self.cfg_ini = base + ".ini"
        os.rename(thisFile2, self.cfg_ini)
        
    def set_trial(self, trial_name):
        '''
            set the trial we want to play
        '''
        self.trial_name = "PlayerPrefsStrial," + trial_name
        
    def set_parameters(self,parameters = ['Port=5005', 'Port=5004']):
        '''
            modify the new configuration file (.ini version) according to our setting and save it
        '''
        original = parameters[0]
        modified = parameters[1]
        
        # Open configuration.ini
        with open(self.cfg_ini, 'r') as file :
          filedata = file.read()
        # Replace the target string
        filedata = filedata.replace(original, modified)
        '''idx = filedata.index('Port=')
        idx_end = -1
        for i in range(idx, len(filedata)):
            if filedata[]
            '''
        # Write the file out again
        with open(self.cfg_ini, 'w') as file:
          file.write(filedata)
        
    def set_local_host(self, IP_address='127.0.0.1',Port=5005):
        '''
            set IP_address for the TCP client. \
            Enter ipconfig in ./cmd for windows machines or ifconfig in bash for linux machines
            to get the local IP address for the unity program. Usually, it is either 192.168.xxx.xxx or 10.xx.xx.xx
        '''
        self.IP_address = IP_address
        self.Port = Port
        
        
    # ------------------------------------------------------------------------------------------------------
    # Helper function for decoding state information
    def decode(self,message):
        return message.decode('utf-8')

    # Helper function for parsing messages
    def parse_message(self,message):
        # For image messages
        if message[0:len("Image")] == bytes("Image", 'utf8'):
            # Set the image to be grayScale or not (grayScale image reduce parameters and train faster)
            if self.grayScale:
                color_code = cv2.COLOR_BGR2GRAY
            else:
                color_code = cv2.COLOR_BGR2RGB
            # There may be a cleaner way to do this conversion, but this works fast enough for comfort
            return {'type':"Image", 
                    'value': cv2.cvtColor(plt.imread(io.BytesIO(message[len("Image"):]), format='JPG'), color_code)}
        elif message[0:len("State")] == bytes("State", 'utf8'):
            # If we receive state information, decode it
            return {'type':"State", 
                    'value': self.decode(message[len("State"):])}
        elif message[0:len("Reward")] == bytes("Reward",'utf8'):
            # If we receive
            #print(self.decode(message[len("Reward"):]))
            return {'type':"Reward", 
                    'value': float(self.decode(message[len("Reward"):]))}
        elif message[0:len("Score")] == bytes("Score",'utf8'):
            # If we receive
            return {'type':"Score", 
                    'value': float(self.decode(message[len("Score"):]))}
        elif message[0:len("EpisodeEnd")] == bytes("EpisodeEnd",'utf8'):
            # If we receive
            #print(self.decode(message[len("EpisodeEnd"):]))
            return {'type':"EpisodeEnd", 
                    'value': self.decode(message[len("EpisodeEnd"):])}
        else:
            # Default messages are returned as-is (with type as None and value as message)
            return {'type': None, 
                    'value': message}
        
    # Below are methods talk between env and client
    #-----------------------------------------------------------------------------------------------------    
    def start(self,grayScale=True,):
        '''
            start a game environment.
            params:
                IP_address: the local TCP/IP of the unity game window. Windows->ipconfig, Linux/Mac->ifconfig
                trial_name: the name of the experiment
        '''
        # Step 1. set grayScale (boolean)
        self.grayScale = grayScale
        print('Port being used: {}'.format(self.Port))
        
        # Step 2. set Port by using set_parameters()
        self.set_parameters(parameters=['Port=5005','Port='+str(self.Port)])
        
        try:
            Popen(self.exe_location)
        except:
            print('Error: cannot start an environment.')
        
        # Step 3. 8 seconds delay to make sure the program is fully set up
        time.sleep(8) 
        #-----------------------------------------------------------------------------------------------------
        # Step 4. TCP Socket Parameters
        TCP_IP = self.IP_address
        TCP_PORT = self.Port
        # End Message Token
        END_TOKEN = "<EOF>"
        
        #print('connection start')
        # Step 5. Set up Socket
        print('local host--{}:{}'.format(TCP_IP,TCP_PORT))
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((TCP_IP, TCP_PORT))
        s.setblocking(0)
        #print('connection established')
        self.s = s
        self.time_out = 100
        # Step 6. call reset_cfg() to make sure 'Port 5005' remains intact so that the next agent is able to call Step 2 successfully
        self.reset_cfg()
        #--------------------------------------------------------------------------------------------------
                                                                                 
    def start_trial(self):
        '''
            strat trial by trial_name and Scene1
        '''
        END_TOKEN = "<EOF>"
        # Set Start Trial Option
        self.s.send(bytes(self.trial_name + END_TOKEN, 'utf8'))
        # Start Scene
        self.s.send(bytes("Scene1" + END_TOKEN, 'utf8'))
                                               
    def end_trial(self):
        '''
            end trial by going back to menue
        '''
        END_TOKEN = "<EOF>"
        self.s.send(bytes("Scene0" + END_TOKEN, 'utf8'))
        
    def new_episode(self):
        '''
            use socket to send "press space" to resume Scene1
        '''
        END_TOKEN = "<EOF>"
        # "Press" Space to continue a another trial
        self.s.send(bytes("PauseSpace" + END_TOKEN, 'utf8'))
        
        
    def get_screenImage(self):
        '''
            return status:
                    image
                    reward
                    time elapsed
                    total score
        '''
        # Data Buffer
        BUFFER_SIZE = 1024
        data = b''
        # End Message Token
        END_TOKEN = "<EOF>"
        
        self.s.send(bytes("ImageRequest" + END_TOKEN, 'utf8'))
        
        start = time.time()
        end = time.time()
        while (end-start)<self.time_out:
            # Wait for data and store in buffer
            ready = select.select([self.s], [], [], 0)
            if ready[0]:
                try:
                    data += self.s.recv(BUFFER_SIZE)
                except ConnectionResetError:
                    print("The connection was closed.")
                    break

                # Check if buffer has a full message
                if(bytes(END_TOKEN, 'utf8') in data):
                    # Decode the message and clear it from the buffer
                    idx = data.index(bytes(END_TOKEN, 'utf8'))
                    while idx != -1:
                        parsed_message = self.parse_message(message=data[0:idx])
                        if parsed_message['type'] == 'Image':
                            # return ScreenImage
                            return parsed_message['value']
            end = time.time()
        print("Doesn't read image as input.")
                            
    def is_episode_finished(self):
        '''
            return True: the episode (trial) is finished, otherwise return False
        '''
        # Data Buffer
        BUFFER_SIZE = 1024
        data = b''
        # End Message Token
        END_TOKEN = "<EOF>"
        
        self.s.send(bytes("EpisodeEnd" + END_TOKEN, 'utf8'))
        
        start = time.time()
        end = time.time()
        while (end-start)<self.time_out:
            # Wait for data and store in buffer
            ready = select.select([self.s], [], [],0)
            if ready[0]:
                try:
                    data += self.s.recv(BUFFER_SIZE)
                except ConnectionResetError:
                    print("The connection was closed.")
                    break

                # Check if buffer has a full message
                if(bytes(END_TOKEN, 'utf8') in data):
                    # Decode the message and clear it from the buffer
                    idx = data.index(bytes(END_TOKEN, 'utf8'))
                    while idx != -1:
                        parsed_message = self.parse_message(message=data[0:idx])
                        
                        if parsed_message['type'] == 'EpisodeEnd':
                            # return ScreenImage
                            if parsed_message['value'] == 'True':
                                return True
                            else:
                                return False
            end = time.time()
        print("Doesn't read episode info.")
        
    def make_action(self,direc,magni):
        '''
           set message as "Rotate122.5" for example.
        '''
        direc = str(direc)
        magni = str(magni)
        
        # End Message Token
        END_TOKEN = "<EOF>"
        
        if np.random.rand() < 0.7:
            self.s.send(bytes("Rotate250" + END_TOKEN, 'utf8'))
        else:
            self.s.send(bytes("Rotate050" + END_TOKEN, 'utf8'))
    
    
    def get_reward(self):
        '''
            get reward feedback from the enviroment
        '''
        # Data Buffer
        BUFFER_SIZE = 1024
        data = b''
        # End Message Token
        END_TOKEN = "<EOF>"
        self.s.send(bytes("Reward" + END_TOKEN, 'utf8'))
        
        start = time.time()
        end = time.time()
        while (end-start)<self.time_out:
            # Wait for data and store in buffer
            ready = select.select([self.s], [], [], 0)
            if ready[0]:
                try:
                    data += self.s.recv(BUFFER_SIZE)
                except ConnectionResetError:
                    print("The connection was closed.")
                    break

                # Check if buffer has a full message
                if(bytes(END_TOKEN, 'utf8') in data):
                    # Decode the message and clear it from the buffer
                    idx = data.index(bytes(END_TOKEN, 'utf8'))
                    while idx != -1:
                        parsed_message = self.parse_message(message=data[0:idx])
                        if parsed_message['type'] == 'Reward':
                            # return ScreenImage
                            return parsed_message['value']
            end = time.time()
        print("Doesn't read reward.")
        
    def get_score(self):
        '''
            get score feedback from the enviroment
        '''
        # Data Buffer
        BUFFER_SIZE = 1024
        data = b''
        # End Message Token
        END_TOKEN = "<EOF>"
        self.s.send(bytes("Score" + END_TOKEN, 'utf8'))
        
        start = time.time()
        end = time.time()
        while (end-start)<self.time_out:
            # Wait for data and store in buffer
            ready = select.select([self.s], [], [], 0)
            if ready[0]:
                try:
                    data += self.s.recv(BUFFER_SIZE)
                except ConnectionResetError:
                    print("The connection was closed.")
                    break

                # Check if buffer has a full message
                if(bytes(END_TOKEN, 'utf8') in data):
                    # Decode the message and clear it from the buffer
                    idx = data.index(bytes(END_TOKEN, 'utf8'))
                    while idx != -1:
                        parsed_message = self.parse_message(message=data[0:idx])
                        if parsed_message['type'] == 'Score':
                            # return ScreenImage
                            return parsed_message['value']
            end = time.time()
        print("Doesn't read score.")