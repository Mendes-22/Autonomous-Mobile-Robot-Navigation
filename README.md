# 🤖 AMR Navigation

This repository contains the implementation of an autonomous mobile robot navigation system developed in **Webots**. The project covers the full navigation pipeline, from raw sensor data acquisition to autonomous path planning and execution.

## 📝 About the Project
This system enables an AMR to operate in unknown or dynamic environments. The navigation pipeline is divided into two main phases:
1. **Mapping:** The robot explores the environment using LIDAR to build an occupancy grid map.
2. **Autonomous Navigation:** Once the map is generated, the user can define any target coordinates in the workspace. The robot then fuses sensor data to track its current pose and executes the A* pathfinding algorithm to reach the destination while dynamically avoiding obstacles.

## 🚀 Core Technologies
* **Mapping (SLAM):** Real-time occupancy grid mapping.
* **Localization:** Extended Kalman Filter (EKF) for sensor fusion.
* **Path Planning:** A* algorithm with dynamic obstacle avoidance.
* **Control:** Proportional navigation for smooth target reaching.

## 🏗️ System Architecture

The project is structured into modular components:

* `kalman_filter.py`: State estimation and sensor fusion for the EKF.
* `path_planner.py`: A* pathfinding engine and trajectory smoothing.
* `SLAM.py`: Controller for SLAM and grid map generation.
* `AMR_main.py`: Controller for path execution and motor control in Webots.

## 🎥 Demo
![Autonomous Navigation Demo](Media/A_Star_SLAM_EKF_Webots_GIF.gif)

