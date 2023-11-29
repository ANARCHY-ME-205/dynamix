cmake_minimum_required(VERSION 3.0.2)
project(dynamix)

find_package(Eigen3 REQUIRED)
find_package(PCL 1.8 REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  pcl_ros
  pcl_conversions
  geometry_msgs
  message_generation
  message_filters
  cv_bridge
  sensor_msgs
  image_geometry
  image_transport
  tf2_ros
  tf2_sensor_msgs
  tf2
)


catkin_package()

###########
## Build ##
###########

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(cluster_extraction src/cluster_extraction.cpp)
target_link_libraries(cluster_extraction ${catkin_LIBRARIES})

add_executable(pointcloud src/depth_pcl.cpp)
target_link_libraries(pointcloud ${catkin_LIBRARIES} ${OpenCV_LIBS})