# Sets the minimum version of CMake required to build the native
# library. You should either keep the default value or only pass a
# value of 3.4.0 or lower.

cmake_minimum_required(VERSION 3.4.1)

aux_source_directory(./src/main/cpp DIR_SRCS)
aux_source_directory(./src/main/cpp/layers DIR_SRCS_LAYERS)
aux_source_directory(./src/main/cpp/test DIR_TEST)
# aux_source_directory(./src/main/cpp/proto DIR_PROTO)
aux_source_directory(./src/main/cpp/msgpack DIR_MSG)

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds it for you.
# Gradle automatically packages shared libraries with your APK.

add_library( # Sets the name of the library.
             MCLdroid-lib

             # Sets the library as a shared library.
             SHARED

             # Provides a relative path to your source file(s).
             # Associated headers in the same location as their source
             # file are automatically included.
             ${DIR_SRCS}
             ${DIR_SRCS_LAYERS}
             ${DIR_TEST}
             ${DIR_MSG}
             )

include_directories(src/main/cpp/include)

# NNPACK_INCLUDE
include_directories(
    /Users/shenjingyuan002/work/SDK/NNPACK/deps/pthreadpool/include
    /Users/shenjingyuan002/work/SDK/NNPACK/deps/fxdiv/include
    /Users/shenjingyuan002/work/SDK/NNPACK/deps/psimd/include
    /Users/shenjingyuan002/work/SDK/NNPACK/deps/fp16/include
    /Users/shenjingyuan002/work/SDK/NNPACK/deps/googlebenchmark/include
)
include_directories(/Users/shenjingyuan002/work/SDK/NNPACK/include)

# PROTOBUF_INCLUDE
#include_directories(/Users/shenjingyuan002/work/SDK/protobuf-261/build4android/include)

# msgpake
include_directories(/Users/shenjingyuan002/work/SDK/msgpack-c-cpp-2.1.1/include)


# Searches for a specified prebuilt library and stores the path as a
# variable. Because system libraries are included in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.

find_library( # Sets the name of the path variable.
              log-lib

              # Specifies the name of the NDK library that
              # you want CMake to locate.
              log )

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in the
# build script, prebuilt third-party libraries, or system libraries.

target_link_libraries( # Specifies the target library.
                       MCLdroid-lib

                       # Links the target library to the log library
                       # included in the NDK.
                       ${log-lib}
                       android
                       jnigraphics
                       #Nnpack
                       /Users/shenjingyuan002/work/NNPACK/lib4android/libnnpack.a
                       /Users/shenjingyuan002/work/NNPACK/lib4android/libnnpack_reference.a
                       /Users/shenjingyuan002/work/NNPACK/lib4android/libnnpack_ukernels.a
                       /Users/shenjingyuan002/work/NNPACK/lib4android/libpthreadpool.a
                       #protobuf
                       #/Users/shenjingyuan002/work/app-repo-android/MCldroid/app/src/main/JniLibs/libprotobuf-lite.a
                      )
