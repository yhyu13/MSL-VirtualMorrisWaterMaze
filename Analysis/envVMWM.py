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
    def __init__(self, config_original_txt_location):
        '''
            initialization & set the path of the original configuraiton text
        '''
        self.cfg_txt = config_original_txt_location
        
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
        os.rename(thisFile2, base + ".ini")
        
    def set_trial(self, trial_name):
        '''
            set the trial we want to play
        '''
        self.trial_name = "PlayerPrefsStrial," + trial_name
        
    def set_parameters(self,parameters):
        '''
            modify the new configuration file (.ini version) according to our setting and save it
        '''
        
        # ONGOING
        
        # TO DO: define all parameters and define corresponding 'if statement'
        ''' 
        import os
        os.rename( afile, afile+"~" )

        destination= open( aFile, "w" )
        source= open( aFile+"~", "r" )
        for line in source:
            destination.write( line )
            if <some condition>:
                destination.write( >some additional line> + "\n" )
        source.close()
        destination.close()
        '''
        
    def set_exe_location(self,VMWMExe_location):
        '''
            set unity program location (.exe)
        '''
        self.exe_location = VMWMExe_location
        
    def set_IP_address(self, IP_address='192.168.56.1'):
        '''
            set IP_address for the TCP client. \
            Enter ipconfig in ./cmd for windows machines or ifconfig in bash for linux machines
            to get the local IP address for the unity program. Usually, it is either 192.168.xxx.xxx or 10.xx.xx.xx
        '''
        self.IP_address = IP_address
        
        
    # ------------------------------------------------------------------------------------------------------
    # Helper function for decoding state information
    def decode_state(self,state_message):
        raise NotImplementedError
        return None

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
                    'value': self.decode_state(message[len("State"):-1])}
        elif message[0:len("Reward")] == bytes("Reward",'utf8'):
            # If we receive
            return {'type':"Reward", 
                    'value': float(message[len("State"):-1])}
        elif message[0:len("Score")] == bytes("Score",'utf8'):
            # If we receive
            return {'type':"Score", 
                    'value': float(message[len("Score"):-1])}
        elif message[0:len("EpisodeEnd")] == bytes("EpisodeEnd",'utf8'):
            # If we receive
            return {'type':"EpisodeEnd", 
                    'value': message[len("EpisodeEnd"):-1]}
        else:
            # Default messages are returned as-is (with type as None and value as message)
            return {'type': None, 
                    'value': message}
        
    def start(self,grayScale=True,humanPlayer=False):
        '''
            start a game environment.
            params:
                IP_address: the local TCP/IP of the unity game window. Windows->ipconfig, Linux/Mac->ifconfig
                trial_name: the name of the experiment
        '''
        self.grayScale = grayScale
        # Part 1 TO DO: open the VMWM.exe
        try:
            Popen(self.exe_location)
        except:
            print('Error: cannot start an environment.')
        # 6 seconds delay to make sure the program is fully set up
        print('Sleep for 10s to ensure fully load up.')
        time.sleep(10) 
        #-----------------------------------------------------------------------------------------------------
        # TCP Socket Parameters
        TCP_IP = self.IP_address # 192.168.56.1 for windows machine
        TCP_PORT = 5005
        # End Message Token
        END_TOKEN = "<EOF>"
        print('connection start')
        # Set up Socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((TCP_IP, TCP_PORT))
        s.setblocking(0)
        print('connection established')
        s.send(bytes("HumanPlayerModefalse" + END_TOKEN, 'utf8'))
        self.s = s
        # Confirm exit
        print('Done')
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
        self.s.send(bytes("PuaseSpace" + END_TOKEN, 'utf8'))
        
        
    def get_screeImage(self):
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
        
        while True:
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
                        else:
                            print("Doesn't read image as input")
                            
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
        while True:
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
                        if parsed_message['type'] == 'EpisodeEnd':
                            # return ScreenImage
                            if parsed_message['value'] == 'True':
                                return True
                            else:
                                return False
                        else:
                            print("Doesn't read paused as input")
        
    def make_action(self,direc,magni):
        '''
            TO DO:
                return message as "Key122.5" for example.
        '''
        direc = str(direc)
        magni = str(magni)
        
        # End Message Token
        END_TOKEN = "<EOF>"
        
        if np.random.rand < 0.7:
            self.s.send(bytes("Key150.0" + END_TOKEN, 'utf8'))
        else:
            self.s.send(bytes("Key050.0" + END_TOKEN, 'utf8'))
    
    
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
        
        while True:
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
                        else:
                            print("Doesn't read reward as input")
        
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
        
        while True:
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
                        else:
                            print("Doesn't read score as input")