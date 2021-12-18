cmake_minimum_required(VERSION 3.10.1)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
project(cuMesh LANGUAGES CXX CUDA)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -frounding-math -std=c++14 -fPIC")


set(CMAKE_CUDA_STANDARD 14)

SET(CUDA_SEPARABLE_COMPILATION ON)
FIND_PACKAGE(CUDA REQUIRED)
if (CMAKE_BUILD_TYPE STREQUAL Debug)
    LIST(APPEND CUDA_NVCC_FLAGS
            -gencode arch=compute_75,code=compute_75
            -gencode arch=compute_75,code=sm_75
            -rdc=true
            --expt-relaxed-constexpr
            -lineinfo
            )
else()
    LIST(APPEND CUDA_NVCC_FLAGS -O3
            -gencode arch=compute_75,code=compute_75
            -gencode arch=compute_75,code=sm_75
            -gencode arch=compute_61,code=compute_61
            -gencode arch=compute_61,code=sm_61
            -rdc=true
            --expt-relaxed-constexpr
            -lineinfo
            )
endif()


#hide variable as SDK now (usually) part of CUDA Toolkit
mark_as_advanced(FORCE CUDA_SDK_ROOT_DIR)
# additional directories where to look for SDK which are
# not yet added in FindCUDA.cmake of CMake
list(APPEND CUDA_SDK_SEARCH_PATH "${CUDA_TOOLKIT_ROOT_DIR}/samples")
# look for required SDK header files as suggested in FindCUDA.cmake
find_path(CUDA_HELPER_INCLUDE_DIR
        helper_cuda.h
        PATHS         ${CUDA_SDK_SEARCH_PATH}
        PATH_SUFFIXES "common/inc"
        DOC           "Location of helper_cuda.h of the CUDA SDK."
        NO_DEFAULT_PATH
        )
find_path(CUDA_HELPER_INCLUDE_DIR helper_cuda.h DOC "Location of helper_cuda.h of the CUDA SDK.")
mark_as_advanced(FORCE CUDA_HELPER_INCLUDE_DIR)
# either add found path to include search path or show user the
# appropriate CMake variables to tell us where to find the headers
if (CUDA_HELPER_INCLUDE_DIR)
    include_directories(${CUDA_HELPER_INCLUDE_DIR})
elseif (NOT CUDA_SDK_ROOT_DIR)
    mark_as_advanced(CLEAR CUDA_SDK_ROOT_DIR)
else ()
    mark_as_advanced(CLEAR CUDA_HELPER_INCLUDE_DIR)
endif ()

find_package(Thrust REQUIRED)
message(STATUS "find thrust in ${THRUST_INCLUDE_DIR}")
include_directories(${THRUST_INCLUDE_DIR})
include_directories(${THRUST_INCLUDE_DIR}/thrust/system/cuda/detail/)

if(NOT MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -std=c++14 -O3 -ffast-math -fPIC")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -std=c++14 -ffast-math -fPIC -g")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -ffast-math -fPIC")
endif()

add_subdirectory(docs)

include_directories(external/probabilistic-quadrics)

FILE(GLOB CU_UTILS_SRC_IMPL
        cuMesh/*.cu
        cuMesh/**/*.cu
        )
FILE(GLOB CU_UTILS_SRC
        cuMesh/*.cpp
        cuMesh/**/*.cpp)

set(CUDA_LINK_LIBRARIES_KEYWORD PRIVATE)
set( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" "--extended-lambda" )

include_directories(${PROJECT_SOURCE_DIR})
cuda_add_library(cuMesh_Impl ${CU_UTILS_SRC_IMPL} )
target_link_libraries(cuMesh_Impl
        PRIVATE ${CUDA_LIBRARIES}
        PRIVATE ${CUDA_CUDART_LIBRARY}
        PRIVATE ${CUDA_cublas_LIBRARY}
        )


message(STATUS ${CU_UTILS_SRC})
add_library(cuMesh SHARED ${CU_UTILS_SRC} )
target_link_libraries(cuMesh
        PRIVATE cuMesh_Impl
        PRIVATE ${CUDA_LIBRARIES}
        PRIVATE ${CUDA_CUDART_LIBRARY}
        PRIVATE ${CUDA_cublas_LIBRARY}
        )



add_subdirectory(tests)
