cmake_minimum_required(VERSION 3.0)
project(lac CXX C)

message(STATUS "CXX compiler: ${CMAKE_CXX_COMPILER}, version: "
        "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C compiler: ${CMAKE_C_COMPILER}, version: "
        "${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")

if(CMAKE_COMPILER_IS_GNUCXX)
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)
    if (GCC_VERSION VERSION_LESS 4.8)
        message(FATAL_ERROR "A gcc compiler with a version >= 4.8.2 is needed.")
    endif()
else()
    message(FATAL_ERROR "A gcc compiler is needed.")
endif(CMAKE_COMPILER_IS_GNUCXX)

set(CMAKE_CXX_FLAGS "-O4 -g -pipe -W -Wall -Wno-unused-parameter -fPIC -fpermissive -std=gnu++11")

if (IS_ABSOLUTE ${PADDLE_ROOT})
    set(PADDLE_ABS_PATH ${PADDLE_ROOT})
else ()
    get_filename_component(PADDLE_ABS_PATH ${CMAKE_BINARY_DIR}/${PADDLE_ROOT} ABSOLUTE)
endif ()

if (EXISTS ${PADDLE_ABS_PATH})
    message(STATUS "\$PADDLE_ROOT is set to '${PADDLE_ABS_PATH}'")
    include_directories(${PADDLE_ABS_PATH})
    include_directories(${PADDLE_ABS_PATH}/third_party/install/gflags/include)
    include_directories(${PADDLE_ABS_PATH}/third_party/boost)
    include_directories(${PADDLE_ABS_PATH}/third_party/install/glog/include)
    include_directories(${PADDLE_ABS_PATH}/third_party/install/protobuf/include)
    include_directories(${PADDLE_ABS_PATH}/third_party/eigen3)
else ()
    message(FATAL_ERROR "'${PADDLE_ABS_PATH}' does not exist.\nPlease set -DPADDLE_ROOT=/path/to/your/fluid_inference_lib/")
endif ()

set (fluid_lib_path ${PADDLE_ABS_PATH}/paddle/fluid/inference)
if (EXISTS ${fluid_lib_path})
    link_directories(${PADDLE_ABS_PATH}/paddle/fluid/inference)
    link_libraries(paddle_fluid)
else ()
    message(FATAL_ERROR "Fluid inference lib not found.\nPlease set -DPADDLE_ROOT=/path/to/your/fluid_inference_lib")
endif ()

# add mkldnn library if it exitsts
set (mkldnn_lib_path ${PADDLE_ABS_PATH}/third_party/install/mkldnn/lib)
if (EXISTS ${mkldnn_lib_path})
    message(STATUS "MKLDNN lib found.")
    link_directories(${PADDLE_ABS_PATH}/third_party/install/mkldnn/lib)
    link_libraries(mkldnn)
else ()
    message(STATUS "MKLDNN lib not found.")
endif()

# add mklml library if it exitsts
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

link_libraries(rt dl pthread)

include_directories(include)
aux_source_directory(src SOURCE)

add_library(lac ${SOURCE} include/ilac.h)
add_executable(lac_demo test/src/lac_demo.cpp)
set_target_properties(lac_demo PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
target_link_libraries(lac_demo lac)
install(TARGETS lac DESTINATION ${PROJECT_SOURCE_DIR}/output/lib)
install(TARGETS lac_demo DESTINATION ${PROJECT_SOURCE_DIR}/output/demo)
install(FILES ${PROJECT_SOURCE_DIR}/include/ilac.h
		DESTINATION ${PROJECT_SOURCE_DIR}/output/include)

