# Unity AI Training for Apple Silicon Macs

This repository contains a framework designed specifically for Apple Silicon Macs, enabling the training of Deep Q-Network (DQN) or Actor-Critic models within Unity. It implements a  computer vision preprocessor to allow the application of these models to expand beyond the training environment and into the real world. The goal is to implement a collection of useful computer vision techniques in C++ so they can be easily used in C# and C++ to allow similar vision based models to be trained in Unity and Unreal Engine.

### Topics Explored

- **Curriculum Learning**: Incrementally increasing the difficulty of tasks to improve learning efficiency and effectiveness. 
- **Behavioral Cloning**: Learning from human demonstration to kickstart the model's ability to perform tasks.

### Currently Exploring: Algorithms and techniques to enable the agent to remember and navigate spatial environments.
- **SLAM**: Implementing it in python to prepare architecture for C++ to easily fit in with the existing code. 
- **NERF**: Also going to implement this in python first.

### Overview of Agents


- **Direct Logging to Unity Console**: A nice feature of this library is its ability to log messages directly to the Unity console from within C++ code. This seamless integration facilitates easier debugging and monitoring of the model's performance and behavior during training and inference.
- **Customizable Preprocessor**: The library includes a highly customizable preprocessor for images. This flexibility allows users to use depth data or lidar mock shader and object recognition. The `PreprocessorOptions` class enables easy adjustments to the preprocessing pipeline, enhancing the model's ability to learn from a diverse range of environments and scenarios.

### [`agents.h`](include/agents.h)

Defines the `ActorCriticAgent` and `ClassificationAgent` classes. They include methods for model inference, training, and interaction with the environment.

- **ActorCriticAgent**: Implements the Actor-Critic method, handling the interactions with the environment, storing experiences, and training the model. (still under construction)

- **ClassificationAgent**: Implements a simple classification scheme for bootstrapping the model with behavioral cloning. Used a simple c# unity script to calculate the optimal action based on the current angle and distance from the target and used this is the ground truth label for each step. This helps skip a lot of the uncertainty early on.

### [`preprocessor.h`](include/preprocessor.h)

- **Preprocessor**: Preprocesses images to prep them for agent inference.
- **PreprocessorOptions**: Allows you to choose between depth only, objects only, or both.

# Unity Project Overview

This project consists of several key components that work together to implement a machine learning classifier within a Unity environment. The components include a C# script for interfacing with the classifier (`AgentInterface.cs`), a script for controlling the movement of a subject within the environment (`SubjectMovement.cs`), and a custom shader for processing depth data (`DepthData.shader`). The shader is used to mimic lidar and can be replaced with the preprocessor depth frames. Below is a brief overview of each component.

### [`AgentInterface.cs`](UnityScripts/AgentInterface.cs)

This script is attached to a camera object within Unity and serves multiple purposes:

- Initializes the camera settings and positions it to face a subject object.
- Handles the creation of a classifier instance through P/Invoke calls to an external library (`libagent_lib`).
- Manages logging of events to a file and Unity's console.
- Captures RGB and depth data from the camera's viewpoint, which can be used by the classifier for inference.
- Calculates the current angle and distance between the camera and the subject, which are metrics relevant to the classifier's operation.

  ### [`SubjectMovement.cs`](UnityScripts/SubjectMovement.cs)

This script controls the movement of a subject within the Unity environment. Key features include:

- Random path generation for the subject's movement, ensuring variability in the data captured by the camera.
- Smooth interpolation of the subject's position over time to simulate continuous movement.
- Public methods for resetting the subject's position and obtaining its current position, which are utilized by the `AgentInterface` script.

### [`DepthData.shader`](UnityScripts/DepthData.shader)

A custom Unity shader designed to process depth data captured by the camera. It converts raw depth data from the camera into a linear scale, which is more useful for certain types of machine learning applications. The shader ensures that depth data is normalized and clamped to a maximum depth value, making it suitable for feeding into the classifier.

### Integration

These components are designed to work together within a Unity project to facilitate the development and testing of machine learning classifiers that rely on visual data. The `AgentInterface` script captures and processes data from the environment, the `SubjectMovement` script provides dynamic movement of a target object, and the `DepthData.shader` ensures that depth data is in a format suitable for machine learning applications.



# Building and Running

This project uses CMake for building. The provided `build.sh` script simplifies the process of compiling and setting up the necessary environment for the project. You will need to know the path to libomp (instructions in `build.sh` on how to find the path) and put this as the libomp path in build.sh.
You will also need to put the scripts directory of your unity project in `replace_unity_libs.sh`. Then, between builds, you can run `build.sh` then `replace_unity_libs.sh` and your environment will be ready.


### Build Instructions

1. Ensure you have CMake installed and configured for your system.
2. Open a terminal in the project's root directory.
3. Run the `build.sh` script:

```sh
./build.sh

