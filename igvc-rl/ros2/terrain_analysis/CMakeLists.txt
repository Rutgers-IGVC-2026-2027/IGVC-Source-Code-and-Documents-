cmake_minimum_required(VERSION 3.16)
project(terrain_analysis VERSION 0.1.0 LANGUAGES CXX)

option(TERRAIN_ANALYSIS_BUILD_ROS "Build the ROS 2 Jazzy node (requires ament)" ON)
option(TERRAIN_ANALYSIS_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  add_compile_options(-Wall -Wextra -Wpedantic -Wconversion -Wshadow)
  if(TERRAIN_ANALYSIS_SANITIZERS)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
  endif()
endif()

add_library(terrain_core src/rolling_grid.cpp src/cloud_decoder.cpp)
target_include_directories(terrain_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
add_executable(terrain_csv_replay src/csv_replay.cpp)
target_link_libraries(terrain_csv_replay PRIVATE terrain_core)

option(TERRAIN_ANALYSIS_BUILD_BENCHMARKS "Build standalone synthetic benchmark tools" OFF)
if(TERRAIN_ANALYSIS_BUILD_BENCHMARKS)
  find_package(Threads REQUIRED)
  add_executable(terrain_benchmark benchmarks/terrain_benchmark.cpp)
  target_link_libraries(terrain_benchmark PRIVATE terrain_core Threads::Threads)
  add_executable(terrain_temporal_benchmark benchmarks/terrain_temporal.cpp)
  target_link_libraries(terrain_temporal_benchmark PRIVATE terrain_core)
endif()

include(CTest)
if(BUILD_TESTING)
  add_executable(terrain_core_tests tests/core_tests.cpp)
  target_link_libraries(terrain_core_tests PRIVATE terrain_core)
  foreach(case ramp obstacle sparse rolling expiry invalid decoder bounded)
    add_test(NAME terrain_${case} COMMAND terrain_core_tests ${case})
  endforeach()
  add_test(NAME terrain_csv_replay COMMAND ${CMAKE_COMMAND}
    -DREPLAY=$<TARGET_FILE:terrain_csv_replay>
    -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
    -DBINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}
    -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/replay_test.cmake)
endif()

install(TARGETS terrain_core terrain_csv_replay EXPORT terrain_core_targets
  ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})
install(FILES README.md LICENSE NOTICE.md DESTINATION share/${PROJECT_NAME})
install(FILES evidence/upstream.json DESTINATION share/${PROJECT_NAME}/evidence)
if(TERRAIN_ANALYSIS_BUILD_ROS)
  find_package(ament_cmake REQUIRED)
  find_package(rclcpp REQUIRED)
  find_package(rcl_interfaces REQUIRED)
  find_package(sensor_msgs REQUIRED)
  find_package(nav_msgs REQUIRED)
  add_executable(terrain_analysis_node src/terrain_node.cpp)
  target_link_libraries(terrain_analysis_node terrain_core)
  ament_target_dependencies(terrain_analysis_node rclcpp rcl_interfaces sensor_msgs nav_msgs)
  install(TARGETS terrain_analysis_node DESTINATION lib/${PROJECT_NAME})
  option(TERRAIN_ANALYSIS_ROS_MESSAGE_TESTS "Run isolated real ROS pub/sub tests on domain 85" OFF)
  if(BUILD_TESTING AND TERRAIN_ANALYSIS_ROS_MESSAGE_TESTS)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(NAME terrain_ros_messages COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/ros_message_tests.py
      --binary $<TARGET_FILE:terrain_analysis_node>
      --output ${CMAKE_CURRENT_BINARY_DIR}/ros-message-results)
    set_tests_properties(terrain_ros_messages PROPERTIES
      ENVIRONMENT "ROS_DOMAIN_ID=85" TIMEOUT 90 RUN_SERIAL TRUE)
  endif()
  ament_export_targets(terrain_core_targets HAS_LIBRARY_TARGET)
  ament_export_include_directories(include)
  ament_package()
else()
  install(EXPORT terrain_core_targets NAMESPACE terrain_analysis:: DESTINATION lib/cmake/terrain_analysis)
endif()
