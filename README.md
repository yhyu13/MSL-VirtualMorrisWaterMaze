# MSL-VirtualMorrisWaterMaze
This repository contains the Unity project for the Memory Systems Lab version of the Virtual Morris Water Maze.


![mazeV0.23.PNG](http://upload-images.jianshu.io/upload_images/1873837-cd7fcce8963d40ef.PNG?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

# Instruction
Download .zip file from the **Release** of thie repo. Then unzip.

*(Note I use tensorflow-gpu. The cpu .yml file is available to you as well)*

Install environment dependencies:
```
conda env create -f Maze.yml
activate Maze
```

Before start testing result, head to Analysis/model.py. Change exe_location and cfg_location to your local repository.

Then download the pretrained model from: [google drive/model](https://drive.google.com/drive/folders/0B6zgGDAEaICRcXIzRTZaTWpFNkE?usp=sharing)

Run a test trail with 15000 eps trained model. By default, the algo will start 6s later after the Unity window launched.
```
python main.py --num_worker=1 --load_model
```

The following discprition is cited from our ongoing publication.

-----

# Abstract
We show a baseline agent for solving the Virtual Morris Water Maze task, using multiple actors with their own environments to simultaneously collect experience. The agent uses contemporary computer vision and memory dependency network. We test the model through variety of scenarios that successfully find the hidden platform fast and reliably. We also found the agent is able to outperform human testers in terms of distance travel, time elapsed and the complexity of path planning through enough training.

# Introduction
In recent years, a variety of reinforcement learning algorithms have been applied to solve more complicated interaction environment than toy games. We are inspired to connect the research between Neuroscience and Machine Learning because Neuroscience heavily inspires reinforcement learning as the concept of temporal difference and sparks Neural Network architecture such as convolutional and recurrent network. The classic experiment of Morris Water Maze (MWM) favors reinforcement learning especially because the following reasons: 
* (1) it requires learning from interaction, 
* (2) it exhibits memory-based real time decision-making problem, 
* (3) no clear optimal path searching strategy is given by a external supervisor. 

Two advantages of MWM are: (1) the experimental development is matured with plenty of setup and measurement technique, (2) researchers have developed Virtual Morris Water Maze (A 3D environment powered by Unity 3D) where we can encapsulate as a reinforcement learning environment. 

Among the reinforcement learning algorithms, Asynchronous Advantage Actor-Critic (A3C) outperforms the previous state-of-the-art algorithm due to its parallelism nature. It takes advantages of multiple cores equipped by most modern computers to train faster and explore more. Another algorithm of interest is Recurrent Deterministic Policy Gradient (RDPG) which is a improvement to solve memory-based control problem in continuous action space. 

# Development

The communication between the Unity environment and the python script is built on Socket. An actor talks to an environment via a IP address (in my case 127.0.0.1) with a most available port picked by Socket. Each actor asynchronously interact, collect data, and update its local operators at the beggining of each episode. The tensorflow graph takes in charge of the asynchronous policy gradient update.

![Capture2.PNG](http://upload-images.jianshu.io/upload_images/1873837-521ee6966b12fa1c.PNG?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

# Architecture

![A3C Neural network.png](http://upload-images.jianshu.io/upload_images/1873837-3bb81c0f50f67140.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

# Result

![Capture4.PNG](http://upload-images.jianshu.io/upload_images/1873837-21dde9a2e827589e.PNG?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![3750train1.png](http://upload-images.jianshu.io/upload_images/1873837-3942345805f426d2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![Capture3.PNG](http://upload-images.jianshu.io/upload_images/1873837-f706cea38ca5d723.PNG?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![Capture3.PNG](http://upload-images.jianshu.io/upload_images/1873837-f706cea38ca5d723.PNG?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

# Ackonwledgement

Thanks to Kevin Horecka at UIUC Beckman Institute for establishing the orignal work, and for his generous support and insightful comments. Thanks to Arthur Juliani for offering greate introduction to A3C architecture:
