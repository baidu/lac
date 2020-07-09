cmake_minimum_required(VERSION 3.0)
project(lac CXX C)

option(WITH_STATIC_LIB "Compile demo with static/shared library, default use static."  ON)

option(WITH_DEMO "Compile C++ demo or not, default yes" ON)
option(WITH_JNILIB "Compile jni library for Java or not, default not" OFF)

# set paddle and java path
#set(PADDLE_ROOT "D:/lac/fluid_inference_install_dir")
#set(JAVA_HOME "C:/Program Files/Java/jdk1.8.0_121")

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
macro(safe_set_static_flag)
    foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
      if(${flag_var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
      endif(${flag_var} MATCHES "/MD")
    endforeach(flag_var)
endmacro()

if(NOT DEFINED PADDLE_ROOT)
    message(FATAL_ERROR "please set PADDLE_ROOT with -DPADDLE_ROOT=/path/paddle/lib")
endif()
if (WITH_JNILIB AND NOT DEFINED JAVA_HOME)
    message(FATAL_ERROR "please set JAVA_HOME with -DJAVA_HOME=/path/of/java")
endif()

if (IS_ABSOLUTE ${PADDLE_ROOT})
    set(PADDLE_ABS_PATH ${PADDLE_ROOT})
else ()
    get_filename_component(PADDLE_ABS_PATH ${CMAKE_BINARY_DIR}/${PADDLE_ROOT}/ ABSOLUTE)
endif ()

message(STATUS "paddle include: ${PADDLE_ABS_PATH}/paddle/include") 
message(STATUS "paddle lib: ${PADDLE_ABS_PATH}/paddle/lib")
include_directories(${PADDLE_ABS_PATH}/paddle/include)
set(PADDLE_LIB_THIRD_PARTY_PATH "${PADDLE_ABS_PATH}/third_party/install/")
include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}protobuf/include")
include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}glog/include")
include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}gflags/include")
include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}xxhash/include")

link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}protobuf/lib")
link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}glog/lib")
link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}gflags/lib")
link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}xxhash/lib")
link_directories("${PADDLE_ABS_PATH}/paddle/lib")


#link_libraries(paddle_fluid)

if (WIN32)
  add_definitions("/DGOOGLE_GLOG_DLL_DECL=")
  if (MSVC)
    option(MSVC_STATIC_CRT "use static C Runtime library by default" ON)
    set(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG} /bigobj /MTd")
    set(CMAKE_C_FLAGS_RELEASE  "${CMAKE_C_FLAGS_RELEASE} /bigobj /MT")
    set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} /bigobj /MTd")
    set(CMAKE_CXX_FLAGS_RELEASE   "${CMAKE_CXX_FLAGS_RELEASE} /bigobj /MT")
    if (WITH_STATIC_LIB)
      safe_set_static_flag()
      add_definitions(-DSTATIC_LIB)
    endif()
  endif()
else()
    set(CMAKE_CXX_FLAGS "-O3 -g -pipe -W -Wall -Wno-unused-parameter -fPIC -fpermissive -std=gnu++11")
endif()


# add mkldnn library if it exitsts
set (mkldnn_inc_path ${PADDLE_ABS_PATH}/third_party/install/mkldnn/include)
set (mkldnn_lib_path ${PADDLE_ABS_PATH}/third_party/install/mkldnn/lib)
if (EXISTS ${mkldnn_inc_path} AND EXISTS ${mkldnn_lib_path})
    message(STATUS "MKLDNN lib found.")
    include_directories(${PADDLE_ABS_PATH}/third_party/install/mkldnn/include)
    if (WIN32)
        set (MKLDNN_LIB ${mkldnn_lib_path}/mkldnn.lib)
    else (WIN32)
        set (MKLDNN_LIB ${mkldnn_lib_path}/libmkldnn.so.0)
    endif (WIN32)
else ()
    message(STATUS "MKLDNN lib not found.")
endif()

# add mklml library if it exists
set (mklml_inc_path ${PADDLE_ABS_PATH}/third_party/install/mklml/include)
set (mklml_lib_path ${PADDLE_ABS_PATH}/third_party/install/mklml/lib)
if (EXISTS ${mklml_inc_path} AND EXISTS ${mklml_lib_path})
    message(STATUS "MKLML lib found.")   
    include_directories(${PADDLE_ABS_PATH}/third_party/install/mklml/include)
    if(WIN32)
        set(MATH_LIB ${mklml_lib_path}/mklml${CMAKE_STATIC_LIBRARY_SUFFIX}
            ${mklml_lib_path}/libiomp5md${CMAKE_STATIC_LIBRARY_SUFFIX})
    else()
        set(MATH_LIB ${mklml_lib_path}/libmklml_intel${CMAKE_SHARED_LIBRARY_SUFFIX}
            ${mklml_lib_path}/libiomp5${CMAKE_SHARED_LIBRARY_SUFFIX})
    endif()
else ()
    message(STATUS "MKLML lib not found.")
endif()

# add openblas library if it exists
set (openblas_inc_path ${PADDLE_ABS_PATH}/third_party/install/openblas/include)
set (openblas_lib_path ${PADDLE_ABS_PATH}/third_party/install/openblas/lib)
if (EXISTS ${openblas_inc_path} AND EXISTS ${openblas_lib_path})
    message(STATUS "openblas lib found.")   
    include_directories(${PADDLE_ABS_PATH}/third_party/install/openblas/include)
    if(WIN32)
        set(MATH_LIB ${openblas_lib_path}/openblas${CMAKE_STATIC_LIBRARY_SUFFIX})
    else()
        set(MATH_LIB ${openblas_lib_path}/libopenblas${CMAKE_STATIC_LIBRARY_SUFFIX})
    endif()
else ()
    message(STATUS "openblas lib not found.")
endif()

if(WITH_STATIC_LIB)
    set(DEPS ${PADDLE_ABS_PATH}/paddle/lib/libpaddle_fluid${CMAKE_STATIC_LIBRARY_SUFFIX})
else()
    if(WIN32)
        set(DEPS ${PADDLE_ABS_PATH}/paddle/lib/paddle_fluid${CMAKE_SHARED_LIBRARY_SUFFIX})
    else()
        set(DEPS ${PADDLE_ABS_PATH}/paddle/lib/libpaddle_fluid${CMAKE_SHARED_LIBRARY_SUFFIX})
    endif()
endif()

if (NOT WIN32)
  set(EXTERNAL_LIB "-ldl -lpthread")
  set(DEPS ${DEPS}
      ${MATH_LIB} ${MKLDNN_LIB}
      glog gflags protobuf  xxhash
      ${EXTERNAL_LIB})
else()
  set(DEPS ${DEPS}
      ${MATH_LIB} ${MKLDNN_LIB}
      glog gflags_static libprotobuf  xxhash ${EXTERNAL_LIB})
  set(DEPS ${DEPS} libcmt shlwapi.lib)
endif(NOT WIN32)


include_directories(c++/include)
aux_source_directory(c++/src SOURCE)


message(STATUS "'${PROJECT_SOURCE_DIR}' output dir")
add_library(lac ${SOURCE})

# for c++ binary demo
if (WITH_DEMO)
add_executable(lac_demo c++/lac_demo.cpp)
set_target_properties(lac_demo PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
target_link_libraries(lac_demo lac ${DEPS})

add_executable(lac_multi c++/lac_multi.cpp)
set_target_properties(lac_multi PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
target_link_libraries(lac_multi lac ${DEPS})
endif()

# for jni lib
if (WITH_JNILIB)
include_directories(./java/cpp ${JAVA_HOME}/include ${JAVA_HOME}/include/linux/ ${JAVA_HOME}/include/darwin ${JAVA_HOME}/include/win32)
add_library(lacjni SHARED ${SOURCE} ./java/cpp/lac_jni.cpp)
set_target_properties(lacjni PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
target_link_libraries(lacjni ${DEPS})
endif()


if(WIN32)
  if (EXISTS ${mklml_inc_path} AND EXISTS ${mklml_lib_path})
    add_custom_command(TARGET lac_demo POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy ${mklml_lib_path}/mklml.dll ${CMAKE_BINARY_DIR}/Release
          COMMAND ${CMAKE_COMMAND} -E copy ${mklml_lib_path}/libiomp5md.dll ${CMAKE_BINARY_DIR}/Release
          COMMAND ${CMAKE_COMMAND} -E copy ${mkldnn_lib_path}/mkldnn.dll  ${CMAKE_BINARY_DIR}/Release
    )
  else()
    add_custom_command(TARGET lac_demo POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy ${openblas_lib_path}/openblas.dll ${CMAKE_BINARY_DIR}/Release
    )
  endif()
  if(NOT WITH_STATIC_LIB)
      add_custom_command(TARGET lac_demo POST_BUILD 
        COMMAND ${CMAKE_COMMAND} -E copy "${PADDLE_ABS_PATH}/paddle/lib/paddle_fluid.dll" ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}
      )
  endif()
endif()


install(TARGETS lac DESTINATION ${PROJECT_SOURCE_DIR}/output/lib)
install(FILES ${PROJECT_SOURCE_DIR}/c++/include/lac.h
        DESTINATION ${PROJECT_SOURCE_DIR}/output/include)


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


