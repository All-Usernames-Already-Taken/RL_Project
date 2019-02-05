# Deep_Q_Routing

This uses deep neural networks to learn the optimal routing paths for a dynamic optical wavelength 5G front-haul computer network.

The project directory is organized in several sub-folders and a stand-alone python file which is the essence of the project. Descriptions of the folders and in some cases their files follow:

agents: contains 3 versions of RL agent objects. Neural networks are located herein.
envs: contains the definition of the objects used to simulate the computer network dynamics
input_data: contains text files that provide the agents with parameters and the network environment structure.
output_data: contains the original version of code used to display results as well as another version used to display results.
project_yml: defines the virtual environment; conda is used
utilities: contains functions that are used in do_learning.py
do_learning.py: simulates a real 5G fronthaul dynamic optical wavelength computer network.
