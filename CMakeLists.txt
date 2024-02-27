cmake_minimum_required(VERSION 3.14)
project(quat_walk)

set(SOURCE_FILES src/quaternions.cpp )

add_executable(render      ${SOURCE_FILES} src/main_render.cpp      extern/glad/src/gl.c)
add_executable(compute     ${SOURCE_FILES} src/main_compute.cpp     extern/glad/src/gl.c)
add_executable(discrepancy ${SOURCE_FILES} src/main_discrepancy.cpp)
add_executable(compute_vk  ${SOURCE_FILES} src/main_vulkan.cpp)

set_target_properties(render      PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
set_target_properties(discrepancy PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
set_target_properties(compute     PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
set_target_properties(compute_vk  PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)

target_include_directories(render      PRIVATE ${CMAKE_SOURCE_DIR} extern/glad/include)
target_include_directories(compute     PRIVATE ${CMAKE_SOURCE_DIR} extern/glad/include)
target_include_directories(compute_vk  PRIVATE ${CMAKE_SOURCE_DIR})
target_include_directories(discrepancy PRIVATE ${CMAKE_SOURCE_DIR})


add_subdirectory(extern/glfw)
target_link_libraries(render      PRIVATE glfw)
target_link_libraries(compute     PRIVATE glfw)

if(WIN32)
    target_link_libraries(render  PRIVATE opengl32)
    target_link_libraries(compute PRIVATE opengl32)
endif()

set(VULKAN_SDK $ENV{VULKAN_SDK})
include_directories(${VULKAN_SDK}/Include)
add_subdirectory(extern/volk)
target_link_libraries(compute_vk  PRIVATE glfw volk)

