cmake_minimum_required(VERSION 3.0)
project(lac CXX C)

option(WITH_DEMO "select to compile C++ binary demo" ON)
option(WITH_JNILIB "select to compile jni library for Java" OFF)

# set(PADDLE_ROOT "./fluid_inference/")
# set(JAVA_HOME "/Library/Java/JavaVirtualMachines/jdk1.8.0_192.jdk/Contents/Home/")

message(STATUS "CXX compiler: ${CMAKE_CXX_COMPILER}, version: "
        "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C compiler: ${CMAKE_C_COMPILER}, version: "
        "${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
#[[
if(CMAKE_COMPILER_IS_GNUCXX)
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)
    if (GCC_VERSION VERSION_LESS 8.2)
        message(FATAL_ERROR "A gcc compiler with a version >= 8.2 is needed.")
    endif()
else()
    message(FATAL_ERROR "A gcc compiler is needed.")
endif(CMAKE_COMPILER_IS_GNUCXX)
]]

set(CMAKE_CXX_FLAGS "-O3 -g -pipe -W -Wall -Wno-unused-parameter -fPIC -fpermissive -std=gnu++11")

if (IS_ABSOLUTE ${PADDLE_ROOT})
    set(PADDLE_ABS_PATH ${PADDLE_ROOT})
else ()
    get_filename_component(PADDLE_ABS_PATH ${CMAKE_BINARY_DIR}/${PADDLE_ROOT}/ ABSOLUTE)
endif ()

message(STATUS "paddle include: ${PADDLE_ABS_PATH}/paddle/include") 
message(STATUS "paddle lib: ${PADDLE_ABS_PATH}/paddle/lib")
include_directories(${PADDLE_ABS_PATH}/paddle/include)
link_directories(${PADDLE_ABS_PATH}/paddle/lib)
link_libraries(paddle_fluid)

# add mkldnn library if it exitsts
set (mkldnn_inc_path ${PADDLE_ABS_PATH}/third_party/install/mkldnn/include)
set (mkldnn_lib_path ${PADDLE_ABS_PATH}/third_party/install/mkldnn/lib)
if (EXISTS ${mkldnn_inc_path} AND EXISTS ${mkldnn_lib_path})
    message(STATUS "MKLDNN lib found.")
    include_directories(${PADDLE_ABS_PATH}/third_party/install/mkldnn/include)
    add_library(mkldnn SHARED IMPORTED)
    set_property(TARGET mkldnn PROPERTY IMPORTED_LOCATION ${PADDLE_ABS_PATH}/third_party/install/mkldnn/lib/libmkldnn.so.0)
    link_libraries(mkldnn)
else ()
    message(STATUS "MKLDNN lib not found.")
endif()

# add mklml library if it exists
set (mklml_inc_path ${PADDLE_ABS_PATH}/third_party/install/mklml/include)
set (mklml_lib_path ${PADDLE_ABS_PATH}/third_party/install/mklml/lib)
if (EXISTS ${mklml_inc_path} AND EXISTS ${mklml_lib_path})
    message(STATUS "MKLML lib found.")   
    include_directories(${PADDLE_ABS_PATH}/third_party/install/mklml/include)
    link_directories(${PADDLE_ABS_PATH}/third_party/install/mklml/lib)
    link_libraries(mklml_intel iomp5)
else ()
    message(STATUS "MKLML lib not found.")
endif()

link_libraries(dl pthread)
include_directories(c++/include)
aux_source_directory(c++/src SOURCE)

message(STATUS "'${PROJECT_SOURCE_DIR}' output dir")
add_library(lac ${SOURCE})

# for c++ binary demo
if (WITH_DEMO)
add_executable(lac_demo c++/lac_demo.cpp)
set_target_properties(lac_demo PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
target_link_libraries(lac_demo lac)

add_executable(lac_multi c++/lac_multi.cpp)
set_target_properties(lac_multi PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
target_link_libraries(lac_multi lac)
endif()

# for jni lib
if (WITH_JNILIB)
include_directories(./java/cpp ${JAVA_HOME}/include ${JAVA_HOME}/include/linux/ ${JAVA_HOME}/include/darwin)
add_library(lacjni SHARED ${SOURCE} ./java/cpp/lac_jni.cpp)
set_target_properties(lacjni PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
endif()


install(TARGETS lac DESTINATION ${PROJECT_SOURCE_DIR}/output/lib)
install(FILES ${PROJECT_SOURCE_DIR}/c++/include/lac.h
        DESTINATION ${PROJECT_SOURCE_DIR}/output/include)

# paddle lib
# install(FILES ${PADDLE_ABS_PATH}/paddle/lib/* 
#         DESTINATION ${PROJECT_SOURCE_DIR}/output/paddle_lib/)

if (WITH_DEMO)
install(TARGETS lac_demo DESTINATION ${PROJECT_SOURCE_DIR}/output/bin)
install(TARGETS lac_multi DESTINATION ${PROJECT_SOURCE_DIR}/output/bin)
endif()

if (WITH_JNILIB)
install(TARGETS lacjni DESTINATION ${PROJECT_SOURCE_DIR}/output/java)

install(FILES ${PROJECT_SOURCE_DIR}/java/com/baidu/nlp/LAC.java
        DESTINATION ${PROJECT_SOURCE_DIR}/output/java/com/baidu/nlp/)
install(FILES ${PROJECT_SOURCE_DIR}/java/LacDemo.java
        DESTINATION ${PROJECT_SOURCE_DIR}/output/java)
endif()


