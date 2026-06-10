cmake_minimum_required(VERSION 3.8)
project(jetank_detection)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find packages
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# Generate interfaces (action).
# NOTE: rosidl_generate_interfaces internally calls
#   ament_python_install_package(${PROJECT_NAME} PACKAGE_DIR <build-dir>)
# for the generated Python bindings, so we must NOT call
# ament_python_install_package() again with the same project name — that
# would create duplicate CMake targets (CMP0002).
# Instead, install the source Python package files directly to the same
# Python site-packages destination that ament_cmake_python would use.
rosidl_generate_interfaces(${PROJECT_NAME}
  "action/DetectSocks.action"
  "msg/SockCloud.msg"
  "action/SegmentSocks.action"
  DEPENDENCIES vision_msgs std_msgs builtin_interfaces sensor_msgs geometry_msgs
)

# Install the source Python package files alongside the generated bindings.
# ament_get_python_install_dir() populates PYTHON_INSTALL_DIR.
ament_get_python_install_dir(PYTHON_INSTALL_DIR)
install(DIRECTORY jetank_detection/
  DESTINATION "${PYTHON_INSTALL_DIR}/${PROJECT_NAME}"
  FILES_MATCHING
    PATTERN "*.py"
  PATTERN "__pycache__" EXCLUDE
)

# Install the node executable so `ros2 run jetank_detection sock_detector_node` works
install(PROGRAMS
  jetank_detection/sock_detector_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME sock_detector_node
)

# Install the dataset capture helper (`ros2 run jetank_detection capture_frames`)
install(PROGRAMS
  jetank_detection/capture_frames.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME capture_frames
)

# Install launch and config directories
install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # Disable copyright check (no copyright headers in this package)
  set(ament_cmake_copyright_FOUND TRUE)
  # Disable xmllint: requires network access to fetch the XSD schema from
  # download.ros.org which is unavailable in this offline build environment.
  set(ament_cmake_xmllint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_import test/test_import.py)
  ament_add_pytest_test(test_logic test/test_logic.py)
endif()

ament_package()
