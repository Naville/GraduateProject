cmake_minimum_required(VERSION 3.12)
project(L4Project)
option(EXECUTION_MODE "Execution Mode. Avilable Options: CPU/GPU" "CPU")
set(PARALLELSTL_USE_PARALLEL_POLICIES ON CACHE BOOL "enable X functionality" FORCE)
option(GPU_THRUST "Use Thrust instead of SyclParallelSTL for GPU" ON)
option(USE_RANGES "Uses the ranges-v3 library" OFF)
option(NESTED_PAR "Allow Nested Parallel Execution" OFF)
set (CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -g -O0 -D DEBUG -Wall -Wno-writable-strings -fopenmp -fopenmp-simd")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -D NDEBUG -Wall -Wno-writable-strings -fopenmp -fopenmp-simd")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_DEBUG} -Wall -g -O3 -D DEBUG -Wno-writable-strings -fopenmp -fopenmp-simd")
set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_RELEASE} -O3 -D NDEBUG -Wno-writable-strings -fopenmp -fopenmp-simd")

include_directories(${CMAKE_CURRENT_LIST_DIR}/include/)

add_compile_definitions(COMPILETIME=__DATE__ CS1="${CMAKE_CXX_FLAGS}" CS2="" CS3="" CS4="" CS5="" CS6="" CS7="")
IF(NESTED_PAR)
ELSE()
add_compile_definitions(NONESTPARALLEL)
ENDIF()
if(USE_RANGES)
add_compile_definitions(USE_RANGES)
include_directories(${CMAKE_CURRENT_LIST_DIR}/range-v3/include/)
ENDIF()
IF(EXECUTION_MODE STREQUAL "CPU")
add_compile_definitions(CPU)
add_compile_definitions(NPBVERSION="NPB 3.0 CPU ParallelSTL Implementation")
FIND_PACKAGE(OpenMP)
IF(OPENMP_FOUND)
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
ELSE()
MESSAGE("OpenMP Not Found!")
ENDIF()
include(${CMAKE_CURRENT_LIST_DIR}/pstl/CMakeLists.txt)
include_directories(${CMAKE_CURRENT_LIST_DIR}/tbb/include/ ${CMAKE_CURRENT_LIST_DIR}/pstl/include/)
link_libraries(ParallelSTL)
ELSEIF(EXECUTION_MODE STREQUAL "GPU")
add_compile_definitions(GPU)
add_compile_definitions(NPBVERSION="NPB 3.0 GPU Thrust Implementation")
IF(GPU_THRUST)
option(THRUST_HOST_SYSTEM "" CPP)
option(THRUST_DEVICE_SYSTEM_OPTIONS "" CUDA)
add_compile_definitions(USE_THRUST)
include(${CMAKE_CURRENT_LIST_DIR}/thrust/CMakeLists.txt)
include_directories(SYSTEM ${CMAKE_CURRENT_LIST_DIR}/thrust/)
ELSE()
add_compile_definitions(NPBVERSION="NPB 3.0 CPU SyclParallelSTL Implementation")
include(${CMAKE_CURRENT_LIST_DIR}/SyclParallelSTL/CMakeLists.txt)
include_directories(${CMAKE_CURRENT_LIST_DIR}/SyclParallelSTL/include/)
link_libraries(SyclSTL)
ENDIF()
ELSE()
MESSAGE(STATUS "Unknown Execution Mode!")
ENDIF()

set (CMAKE_CXX_STANDARD 17)
file(GLOB BT_SRC
      "include/*.h"
      "BT/*.cpp"
)
file(GLOB CG_SRC
      "include/*.h"
      "CG/*.cpp"
)
file(GLOB FT_SRC
      "include/*.h"
      "FT/*.cpp"
)
file(GLOB CORE_SRC
      "include/*.h"
      "lib/*.c"
)
file(GLOB R3T_SRC
      "include/*.h"
      "r3test/*.cpp"
)
add_library(Core ${CORE_SRC})

# BT
add_executable(BTS ${BT_SRC})
target_link_libraries(BTS Core)
target_compile_definitions(BTS PRIVATE PROBLEM_SIZE=12 NITER_DEFAULT=60 DT_DEFAULT=0.010 CONVERTDOUBLE=FALSE)

add_executable(BTW ${BT_SRC})
target_link_libraries(BTW Core)
target_compile_definitions(BTW PRIVATE PROBLEM_SIZE=24 NITER_DEFAULT=200 DT_DEFAULT=0.0008 CONVERTDOUBLE=FALSE)


add_executable(BTA ${BT_SRC})
target_link_libraries(BTA Core)
target_compile_definitions(BTA PRIVATE PROBLEM_SIZE=64 NITER_DEFAULT=200 DT_DEFAULT=0.0008 CONVERTDOUBLE=FALSE)

add_executable(BTB ${BT_SRC})
target_link_libraries(BTB Core)
target_compile_definitions(BTB PRIVATE PROBLEM_SIZE=102 NITER_DEFAULT=200 DT_DEFAULT=0.0003 CONVERTDOUBLE=FALSE)

# FT

add_executable(FTS ${FT_SRC})
target_link_libraries(FTS Core)
target_compile_definitions(FTS PRIVATE NX=64 NY=64 NZ=64 MAXDIM=64 NITER_DEFAULT=6 NTOTAL=262144 CONVERTDOUBLE=FALSE)

add_executable(FTW ${FT_SRC})
target_link_libraries(FTW Core)
target_compile_definitions(FTW PRIVATE NX=128 NY=128 NZ=32 MAXDIM=128 NITER_DEFAULT=6 NTOTAL=524288 CONVERTDOUBLE=FALSE)

add_executable(FTA ${FT_SRC})
target_link_libraries(FTA Core)
target_compile_definitions(FTA PRIVATE NX=256 NY=256 NZ=128 MAXDIM=256 NITER_DEFAULT=6 NTOTAL=8388608 CONVERTDOUBLE=FALSE)

add_executable(FTB ${FT_SRC})
target_link_libraries(FTB Core)
target_compile_definitions(FTB PRIVATE NX=512 NY=256 NZ=256 MAXDIM=512 NITER_DEFAULT=20 NTOTAL=33554432 CONVERTDOUBLE=FALSE)

# CG
add_executable(CGS ${CG_SRC})
target_link_libraries(CGS Core)
target_compile_definitions(CGS PRIVATE NA=1400 NONZER=7 NITER=15 SHIFT=10.0 RCOND=1.0e-1 CONVERTDOUBLE=FALSE)

add_executable(CGW ${CG_SRC})
target_link_libraries(CGW Core)
target_compile_definitions(CGW PRIVATE NA=7000 NONZER=8 NITER=15 SHIFT=12.0 RCOND=1.0e-1 CONVERTDOUBLE=FALSE)

add_executable(CGA ${CG_SRC})
target_link_libraries(CGA Core)
target_compile_definitions(CGA PRIVATE NA=14000 NONZER=11 NITER=15 SHIFT=20.0 RCOND=1.0e-1 CONVERTDOUBLE=FALSE)

add_executable(CGB ${CG_SRC})
target_link_libraries(CGB Core)
target_compile_definitions(CGB PRIVATE NA=75000 NONZER=13 NITER=75 SHIFT=60.0 RCOND=1.0e-1 CONVERTDOUBLE=FALSE)



add_executable(rangev3ext_test ${R3T_SRC})
target_include_directories(rangev3ext_test PUBLIC ${CMAKE_CURRENT_LIST_DIR}/range-v3/include/)
