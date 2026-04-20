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
find_package(CUDA REQUIRED)
find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)

set(G2O_INCLUDE_DIRS /usr/local/include)
set(G2O_CORE_LIBRARY /usr/local/lib/libg2o_core.so)
set(G2O_STUFF_LIBRARY /usr/local/lib/libg2o_stuff.so)
set(G2O_SOLVER_PCG /usr/local/lib/libg2o_solver_pcg.so)
set(G2O_SOLVER_CSPARSE /usr/local/lib/libg2o_solver_csparse.so)
#set(G2O_SOLVER_CHOLMOD /usr/local/lib/libg2o_solver_cholmod.so)
set(G2O_TYPES_SLAM2D /usr/local/lib/libg2o_types_slam2d.so)
set(G2O_TYPES_SLAM2D_ADDONS /usr/local/lib/libg2o_types_slam2d_addons.so)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler;-fPIC" CACHE STRING "nvcc flags" FORCE)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--ftz=true;--prec-div=false;--prec-sqrt=false")
set(CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc verbose" FORCE)
cuda_add_library(icp_cuda_lib SHARED src/icp_cuda.cu)
set_target_properties(icp_cuda_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

# Add the executable and specify dependencies
add_executable(cmd_vel_to_odom src/cmd_vel_to_odom.cpp)
add_executable(odom_pub src/odom_pub.cpp)
add_executable(sensor_monitor_node src/test.cpp)
add_executable(tick_controller src/tick.cpp)
add_executable(icp_compare_node 
  src/relative.cpp
  src/icp_wrapper.cpp
)

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

ament_target_dependencies(tick_controller
  rclcpp
  std_msgs
  geometry_msgs
)

ament_target_dependencies(icp_compare_node
  rclcpp 
  sensor_msgs
  nav_msgs
  std_msgs
  geometry_msgs
)

target_link_libraries(icp_compare_node
  Eigen3::Eigen
  ${PCL_LIBRARIES}
  ${CUDA_LIBRARIES}
  icp_cuda_lib
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

install(TARGETS
  tick_controller
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS
  icp_compare_node
  DESTINATION lib/${PROJECT_NAME}
)
  
install(
  DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

ament_package()