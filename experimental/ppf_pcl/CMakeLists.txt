cmake_minimum_required(VERSION 3.16)
project(zutik_ppf_bridge LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# The experimental image currently resolves Debian bookworm's official PCL
# 1.13.0 package.  PPFEstimation/PPFRegistration used here are present in that
# release; the bridge does not claim to require the newer 1.14/1.15 API.
find_package(PCL 1.13 REQUIRED COMPONENTS common features registration)

add_executable(zutik_ppf_bridge ppf_bridge.cpp)
target_include_directories(zutik_ppf_bridge PRIVATE ${PCL_INCLUDE_DIRS})
target_compile_definitions(zutik_ppf_bridge PRIVATE ${PCL_DEFINITIONS})
target_link_directories(zutik_ppf_bridge PRIVATE ${PCL_LIBRARY_DIRS})
target_link_libraries(zutik_ppf_bridge PRIVATE ${PCL_LIBRARIES})
