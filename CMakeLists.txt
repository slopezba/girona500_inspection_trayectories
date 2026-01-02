cmake_minimum_required(VERSION 3.0.2)
project(girona500_inspection_trajectories)

## Compile as C++11 (safe default for Noetic)
add_compile_options(-std=c++11)

################################################
## Find catkin macros and libraries
################################################
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  nav_msgs
  visualization_msgs
  std_srvs
  actionlib
  actionlib_msgs
  dynamic_reconfigure
)

################################################
## Dynamic reconfigure
################################################
generate_dynamic_reconfigure_options(
  cfg/PlaneInspection.cfg
  cfg/InspectionController.cfg
)

################################################
## Messages, services and actions
################################################

## Services
add_service_files(
  FILES
  PlanInspectionPath.srv
)

## Actions (para el siguiente paso, aunque aún no la uses)
add_action_files(
  FILES
  ExecutePlaneInspection.action
)

## Generate messages
generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  nav_msgs
  actionlib_msgs
)

################################################
## Catkin package
################################################
catkin_package(
  CATKIN_DEPENDS
    rospy
    std_msgs
    geometry_msgs
    nav_msgs
    visualization_msgs
    std_srvs
    actionlib
    actionlib_msgs
    dynamic_reconfigure
)

################################################
## Install
################################################

## Python scripts
catkin_install_python(
  PROGRAMS
    scripts/plane_inspection_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Launch files
install(
  DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

## Config files (YAML + cfg visible)
install(
  DIRECTORY config cfg
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
