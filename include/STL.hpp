#ifdef CPU
#define PSTL_USE_PARALLEL_POLICIES 1
#include "pstl/execution"
#include "pstl/algorithm"
#include "pstl/numeric"
#include "pstl/memory"
#if !__PSTL_CPP17_EXECUTION_POLICIES_PRESENT
#define PARALLEL __pstl::execution::par
#define PARALLELUNSEQ __pstl::execution::par
#else
#define PARALLEL std::execution::par
#define PARALLELUNSEQ std::execution::par
#endif
#define NS std
#else if defined(GPU)
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
