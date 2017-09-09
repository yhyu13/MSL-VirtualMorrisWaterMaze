# MSL-VirtualMorrisWaterMaze
This repository contains the Unity project for the Memory Systems Lab version of the Virtual Morris Water Maze.


![mazeV0.23.PNG](http://upload-images.jianshu.io/upload_images/1873837-cd7fcce8963d40ef.PNG?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

# abstract
We show a baseline agent for solving the Virtual Morris Water Maze task, using multiple actors with their own environments to simultaneously collect experience. The agent uses contemporary computer vision and memory dependency network. We test the model through variety of scenarios that successfully replicate the behaviors of both rodents and humans in these tasks. We also found the agent is able to outperform human testers in terms of distance travel, time elapsed and the complexity of path planning through enough training.

# Introduction
In recent years, a variety of reinforcement learning algorithms have been applied to solve more complicated interaction environment than toy games. We are inspired to connect the research between Neuroscience and Machine Learning because Neuroscience heavily inspires reinforcement learning as the concept of temporal difference and sparks Neural Network architecture such as convolutional and recurrent network. The classic experiment of Morris Water Maze (MWM) favors reinforcement learning especially because the following reasons: 
* (1) it requires learning from interaction, 
* (2) it exhibits memory-based real time decision-making problem, 
* (3) no clear optimal path searching strategy is given by a external supervisor. 

Other benefits of MWM are: (1) the experimental development is matured with plenty of setup and measurement technique, (2) researchers have developed Virtual Morris Water Maze (A 3D environment powered by Unity 3D) where we can encapsulate as a reinforcement learning environment. 

Among the reinforcement learning algorithms, Asynchronous Advantage Actor-Critic (A3C) outperforms the previous state-of-the-art algorithm due to its parallelism nature. It takes advantages of multiple cores equipped by most modern computers to train faster and explore more. Another algorithm of interest is Recurrent Deterministic Policy Gradient (RDPG) which is a improvement to solve memory-based control problem in continuous action space. 
