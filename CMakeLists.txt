cmake_minimum_required(VERSION 3.5)
project(kwj_localization)

# Use C++14 standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)  

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

# Add the executable and specify dependencies
add_executable(cmd_vel_to_odom src/cmd_vel_to_odom.cpp)
add_executable(odom_pub src/odom_pub.cpp)
add_executable(sensor_monitor_node src/test.cpp)

# Link the necessary ROS2 libraries to the executable
ament_target_dependencies(cmd_vel_to_odom
  rclcpp
  geometry_msgs
  nav_msgs
  sensor_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
)

ament_target_dependencies(odom_pub
  rclcpp
  std_msgs
  geometry_msgs
  nav_msgs
  sensor_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
)

ament_target_dependencies(sensor_monitor_node
  rclcpp
  nav_msgs
  sensor_msgs
)

# Install the executable
install(TARGETS cmd_vel_to_odom
  RUNTIME DESTINATION lib/${PROJECT_NAME})
  
 install(TARGETS odom_pub
  RUNTIME DESTINATION lib/${PROJECT_NAME})
  
 install(TARGETS
  sensor_monitor_node
  DESTINATION lib/${PROJECT_NAME}
)
  
install(
  DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

ament_package()