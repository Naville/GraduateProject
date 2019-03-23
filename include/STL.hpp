#include <vector>
#ifdef USE_RANGES
#include "range.hpp"
#include <range/v3/all.hpp>
#define MAKE_RANGE(i,j,varname) \
  auto varname=view::ints(i,j);
#define MAKE_RANGE_UNDEF(i,j,varname) \
  varname=view::ints(i,j);
#else
#define MAKE_RANGE(i,j,varname) \
  auto varname=std::vector<int>(j-i);\
  std::iota(varname .begin(), varname .end(),i);
#define MAKE_RANGE_UNDEF(i,j,varname) \
  varname.resize(j-i);\
  std::iota(varname .begin(), varname .end(),i);
#endif
/*#include "range.hpp"
#include <range/v3/all.hpp>
#define MAKE_RANGE(i,j,varname) \
  ranges::iota_view varname(i,j);
#define MAKE_RANGE_UNDEF(i,j,varname) \
  varname=ranges::iota_view(i,j);*/
#ifdef CPU
  #include "pstl/execution"
  #include "pstl/algorithm"
  #include "pstl/numeric"
  #include "pstl/memory"
  #if !__PSTL_CPP17_EXECUTION_POLICIES_PRESENT
  #define SEQ __pstl::execution::seq
  #define UNSEQ __pstl::execution::unseq
  #define PARALLEL __pstl::execution::par
  #define PARALLELUNSEQ __pstl::execution::par_unseq
  #else
  #define SEQ std::execution::seq
  #define UNSEQ std:execution::unseq
  #define PARALLEL std::execution::par
  #define PARALLELUNSEQ std::execution::par_unseq
  #endif
  #define NS std
#elif defined(GPU)
  #ifdef USE_THRUST
    #include <thrust/host_vector.h>
    #include <thrust/device_vector.h>
    #include <thrust/generate.h>
    #include <thrust/sort.h>
    #include <thrust/copy.h>
    #define NS thrust
  #else
    #include "sycl/execution_policy"
    #include "experimental/algorithm"
    #define PARALLEL sycl::sycl_execution_policy<>()
    #define PARALLELUNSEQ sycl::sycl_execution_policy<>()
    #define NS std::experimental::parallel
  #endif
#else
  #error Unknown Execution Mode
#endif
#ifdef NONESTPARALLEL
#define NESTPAR SEQ
#define NESTPARUNSEQ UNSEQ
#else
#define NESTPAR PARALLEL
#define NESTPARUNSEQ PARALLELUNSEQ
#endif
