# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.4.1)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/paddle/include)

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.
set(CMAKE_CXX_FLAGS
        "${CMAKE_CXX_FLAGS} -ffast-math -Ofast -Os -DNDEBUG -fno-exceptions -fomit-frame-pointer -fno-asynchronous-unwind-tables -fno-unwind-tables"
        )
set(CMAKE_CXX_FLAGS
        "${CMAKE_CXX_FLAGS} -fvisibility=hidden -fvisibility-inlines-hidden -fdata-sections -ffunction-sections"
        )
set(CMAKE_SHARED_LINKER_FLAGS
        "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--gc-sections -Wl,-z,nocopyreloc")

add_library( # Sets the name of the library.
        native_lib
        # Sets the library as a shared library.
        SHARED
        # Provides a relative path to your source file(s).
        native_lib.cpp
        lac.h
        lac.cpp
        lac_util.h
        lac_util.cpp
        )

find_library( # Sets the name of the path variable.
        log-lib
        # Specifies the name of the NDK library that
        # you want CMake to locate.
        log)

# add paddle library
add_library(
        paddle_lite_jni
        SHARED
        IMPORTED
)

set_target_properties(
        paddle_lite_jni
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libpaddle_lite_jni.so)


target_link_libraries( # Specifies the target library.
        native_lib
        paddle_lite_jni
        # Links the target library to the log library
        # included in the NDK.
        ${log-lib})