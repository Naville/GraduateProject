cmake_minimum_required(VERSION 3.12)
project(L4Project)
option(EXECUTION_MODE "Execution Mode" "CPU")
option(PARALLELSTL_USE_PARALLEL_POLICIES "" ON)
option(TBB_DIR "TBB DIR" ${CMAKE_CURRENT_LIST_DIR}/tbb/)
set (CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -g -O0 -D DEBUG -Wno-writable-strings -fopenmp")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -D NDEBUG -Wno-writable-strings -fopenmp")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_DEBUG} -Wall -g -O3 -D DEBUG -Wno-writable-strings -fopenmp")
set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_RELEASE} -O3 -D NDEBUG -Wno-writable-strings -fopenmp")


IF(EXECUTION_MODE STREQUAL "CPU")
add_compile_definitions(CPU)
FIND_PACKAGE(OpenMP)
IF(OPENMP_FOUND)
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
ELSE()
MESSAGE(FATAL_ERROR "OpenMP Not Found!")
ENDIF()
include(${CMAKE_CURRENT_LIST_DIR}/pstl/CMakeLists.txt)
include_directories(${CMAKE_CURRENT_LIST_DIR}/tbb/include/ ${CMAKE_CURRENT_LIST_DIR}/pstl/include/ ${CMAKE_CURRENT_LIST_DIR}/include/ ${CMAKE_CURRENT_LIST_DIR}/range-v3/include/)
link_libraries(ParallelSTL)
ELSEIF(EXECUTION_MODE STREQUAL "GPU")
add_compile_definitions(GPU)
include(${CMAKE_CURRENT_LIST_DIR}/SyclParallelSTL/CMakeLists.txt)
include_directories(${CMAKE_CURRENT_LIST_DIR}/SyclParallelSTL/include/ ${CMAKE_CURRENT_LIST_DIR}/include/ ${CMAKE_CURRENT_LIST_DIR}/range-v3/include/)
link_libraries(SyclSTL)
ELSE()
MESSAGE(FATAL_ERROR "Unknown Execution Mode!")
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
add_library(Core ${CORE_SRC})
add_executable(BT ${BT_SRC})
add_executable(CG ${CG_SRC})
add_executable(FT ${FT_SRC})
target_link_libraries(BT Core)
target_link_libraries(FT Core)
target_link_libraries(CG Core)
