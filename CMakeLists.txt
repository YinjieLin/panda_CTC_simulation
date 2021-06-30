
# Copyright (c) 2021, Yinjie Lin
# All rights reserved.

cmake_minimum_required(VERSION 3.5)
project(panda_sim_mujoco)
set(CMAKE_CXX_STANDARD 11)

set(USE_GL 1)
set(BIN_NAME ${PROJECT_NAME})

# does not need X11.
option(RUN_X11_TESTS "Run tests that require X11" OFF)
find_package(Eigen3 REQUIRED)
#Mujoco paths
set(MJPRO_PATH /home/yjlin/.mujoco/mujoco200)
include_directories(${MJPRO_PATH}/include)
link_directories(${MJPRO_PATH}/bin)
message(STATUS "MuJoCo path: " ${MJPRO_PATH})

#Finding main mujoco library
if(${USE_GL})
    file(GLOB LIB_MUJOCO ${MJPRO_PATH}/bin/libmujoco[0-9][0-9][0-9].so)
else()
    file(GLOB LIB_MUJOCO ${MJPRO_PATH}/bin/libmujoco[0-9][0-9][0-9]nogl.so)
endif()
#Showing mujoco library found
message(STATUS "MuJoCo lib: " ${LIB_MUJOCO})

add_executable(${BIN_NAME} 
src/my_sim_CTC.cpp
)

target_link_libraries(${BIN_NAME} ${LIB_MUJOCO})

#Standard libraries for GL
target_link_libraries(${PROJECT_NAME} GL GLU glut )

#Additional libraries from mujoco package
target_link_libraries(${PROJECT_NAME} ${MJPRO_PATH}/bin/libglew.so )
target_link_libraries(${PROJECT_NAME} ${MJPRO_PATH}/bin/libglfw.so.3 )
#target_link_libraries(${PROJECT_NAME}_exe libglewegl.so )
#target_link_libraries(${PROJECT_NAME}_exe libglewosmesa.so)

target_include_directories(${PROJECT_NAME} PUBLIC
 include
)