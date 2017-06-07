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
        self.trail_name = "PlayerPrefsStrial," + trial_name
        
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
            # There may be a cleaner way to do this conversion, but this works fast enough for comfort
            return {'type':"Image", 
                    'value': cv2.cvtColor(plt.imread(io.BytesIO(message[len("Image"):]), format='JPG'), cv2.COLOR_BGR2RGB)}
        elif message[0:len("State")] == bytes("State", 'utf8'):
            # If we receive state information, decode it
            return {'type':"State", 
                    'value': self.decode_state(message[len("State"):-1])}
        else:
            # Default messages are returned as-is (with type as None and value as message)
            return {'type': None, 
                    'value': message}
        
    def start(self):
        '''
            start a game environment.
            params:
                IP_address: the local TCP/IP of the unity game window. Windows->ipconfig, Linux/Mac->ifconfig
                trial_name: the name of the experiment
        '''
        
        # ONGOING
        # Part 1 TO DO: open the VMWM.exe
        try:
            Popen(self.exe_location)
        except:
            print('Error: cannot start an environment.')
        # 6 seconds delay to make sure the program is fully set up
        time.sleep(6) 
        #-----------------------------------------------------------------------------------------------------
        
        # ONGING
        
        # Part 2 TO DO: setup connection
        
        # Set up OpenCV Window
        #cv2.startWindowThread()

        # TCP Socket Parameters
        TCP_IP = self.IP_address # 192.168.56.1 for windows machine
        TCP_PORT = 5005
        # Data Buffer
        BUFFER_SIZE = 1024
        data = b''

        # End Message Token
        END_TOKEN = "<EOF>"
        
        print('connection start')
        
        # Set up Socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((TCP_IP, TCP_PORT))
        s.setblocking(0)

        # Set Start Trial Option
        s.send(bytes(self.trail_name + END_TOKEN, 'utf8'))

        # Start Scene
        s.send(bytes("Scene1" + END_TOKEN, 'utf8'))
        
        print('connection established')
        
        # Loop until keyboard interrupt, reading/requesting data
        while True:
            try:
                # Wait for data and store in buffer
                ready = select.select([s], [], [], 0)
                if ready[0]:
                    try:
                        data += s.recv(BUFFER_SIZE)
                    except ConnectionResetError:
                        print("The connection was closed.")
                        break

                # Check if buffer has a full message
                if(bytes(END_TOKEN, 'utf8') in data):
                    # Decode the message and clear it from the buffer
                    idx = data.index(bytes(END_TOKEN, 'utf8'))
                    while idx != -1:
                        parsed_message = self.parse_message(message=data[0:idx])

                        # If the message was an image, display it using OpenCV (faster than matplotlib)
                        if parsed_message['type'] == 'Image':
                            #cv2.imshow('frame', parsed_message['value'])
                            #cv2.waitKey(1)

                            # Send a request for a new frame
                            s.send(bytes("KeyUP" + END_TOKEN, 'utf8'))
                            # s.send(bytes("ImageRequest" + END_TOKEN, 'utf8'))

                        data = data[idx + len(END_TOKEN):-1]
                        if(data != ''):
                            try:
                                idx = str(data).index(END_TOKEN)
                            except ValueError:
                                    break

            # Wait for a keyboard interrupt to stop the loop
            except KeyboardInterrupt:
                break

        # Close the socket and destroy the display window
        #s.close()
        #cv2.destroyAllWindows()
        
        self.env = s

        # Confirm exit
        print('Done')