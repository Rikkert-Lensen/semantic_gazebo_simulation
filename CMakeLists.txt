cmake_minimum_required(VERSION 3.0.2)
project(semantic_gazebo_simulation)

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
  message(STATUS "OPENMP FOUND")
  set(OpenMP_FLAGS ${OpenMP_CXX_FLAGS})  # or if you use C: ${OpenMP_C_FLAGS}
  set(OpenMP_LIBS gomp)
endif()


set(PACKAGE_DEPENDENCIES
  roscpp
  rospy
  std_msgs
  sensor_msgs
  message_runtime
  tf2_ros
  rosbag
  cv_bridge
)

find_package(catkin REQUIRED COMPONENTS
  ${PACKAGE_DEPENDENCIES}
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

catkin_package(
  CATKIN_DEPENDS ${PACKAGE_DEPENDENCIES}
)

set(LINK_LIBS
  ${catkin_LIBRARIES}
  ${OpenMP_LIBS}
)

add_executable(camera_pose_transformer
  src/camera_pose_transformer.cpp
)
target_link_libraries(camera_pose_transformer ${LINK_LIBS})

catkin_install_python(
  PROGRAMS
  scripts/semantic_cloud_publisher.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(TARGETS camera_pose_transformer
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY
  include
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
  )