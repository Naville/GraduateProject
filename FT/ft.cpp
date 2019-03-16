/*--------------------------------------------------------------------

  NAS Parallel Benchmarks 3.0 structured OpenMP C versions - FT

  This benchmark is an OpenMP C version of the NPB FT code.

  The OpenMP C 2.3 versions are derived by RWCP from the serial Fortran versions
  in "NPB 2.3-serial" developed by NAS. 3.0 translation is performed by the
UVSQ.

  Permission to use, copy, distribute and modify this software for any
  purpose with or without fee is hereby granted.
  This software is provided "as is" without express or implied warranty.

  Information on OpenMP activities at RWCP is available at:

           http://pdplab.trc.rwcp.or.jp/pdperf/Omni/

  Information on NAS Parallel Benchmarks 2.3 is available at:

           http://www.nas.nasa.gov/NAS/NPB/

--------------------------------------------------------------------*/
/*--------------------------------------------------------------------

  Authors: D. Bailey
           W. Saphir

  OpenMP C version: S. Satoh

  3.0 structure translation: M. Popov

--------------------------------------------------------------------*/
#ifdef CPU
#define PSTL_USAGE_WARNINGS 1
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
#else
#include "sycl/execution_policy"
#include "experimental/algorithm"
#define PARALLEL sycl::sycl_execution_policy<>()
#define PARALLELUNSEQ sycl::sycl_execution_policy<>()
#define NS std::experimental::parallel
#endif
#include "range.hpp"
#include <range/v3/all.hpp>
/* global variables */
#include "npb-C.h"
#include "global.h"
#include <iostream>
using namespace ranges;

/* function declarations */
void evolve(dcomplex u0[NZ][NY][NX], dcomplex u1[NZ][NY][NX], int t,
                   int indexmap[NZ][NY][NX], int d[3]);
void compute_initial_conditions(dcomplex u0[NZ][NY][NX], int d[3]);
void ipow46(double a, int exponent, double *result);
void setup(void);
void compute_indexmap(int indexmap[NZ][NY][NX], int d[3]);
void print_timers(void);
void fft(int dir, dcomplex x1[NZ][NY][NX], dcomplex x2[NZ][NY][NX]);
void cffts1(int is, int d[3], dcomplex x[NZ][NY][NX],
                   dcomplex xout[NZ][NY][NX], dcomplex y0[NX][FFTBLOCKPAD],
                   dcomplex y1[NX][FFTBLOCKPAD]);
void cffts2(int is, int d[3], dcomplex x[NZ][NY][NX],
                   dcomplex xout[NZ][NY][NX], dcomplex y0[NX][FFTBLOCKPAD],
                   dcomplex y1[NX][FFTBLOCKPAD]);
void cffts3(int is, int d[3], dcomplex x[NZ][NY][NX],
                   dcomplex xout[NZ][NY][NX], dcomplex y0[NX][FFTBLOCKPAD],
                   dcomplex y1[NX][FFTBLOCKPAD]);
void fft_init(int n);
void cfftz(int is, int m, int n, dcomplex x[NX][FFTBLOCKPAD],
                  dcomplex y[NX][FFTBLOCKPAD]);
void fftz2(int is, int l, int m, int n, int ny, int ny1, dcomplex u[NX],
                  dcomplex x[NX][FFTBLOCKPAD], dcomplex y[NX][FFTBLOCKPAD]);
static int ilog2(int n);
void checksum(int i, dcomplex u1[NZ][NY][NX], int d[3]);
void verify(int d1, int d2, int d3, int nt, boolean *verified,
                   char *cls);

/*--------------------------------------------------------------------
c FT benchmark
c-------------------------------------------------------------------*/

int main(int argc, char **argv) {

  /*c-------------------------------------------------------------------
  c-------------------------------------------------------------------*/

  int i, ierr;

  /*------------------------------------------------------------------
  c u0, u1, u2 are the main arrays in the problem.
  c Depending on the decomposition, these arrays will have different
  c dimensions. To accomodate all possibilities, we allocate them as
  c one-dimensional arrays and pass them to subroutines for different
  c views
  c  - u0 contains the initial (transformed) initial condition
  c  - u1 and u2 are working arrays
  c  - indexmap maps i,j,k of u0 to the correct i^2+j^2+k^2 for the
  c    time evolution operator.
  c-----------------------------------------------------------------*/

  /*--------------------------------------------------------------------
  c Large arrays are in common so that they are allocated on the
  c heap rather than the stack. This common block is not
  c referenced directly anywhere else. Padding is to avoid accidental
  c cache problems, since all array sizes are powers of two.
  c-------------------------------------------------------------------*/
  static dcomplex u0[NZ][NY][NX];
  static dcomplex pad1[3];
  static dcomplex u1[NZ][NY][NX];
  static dcomplex pad2[3];
  static dcomplex u2[NZ][NY][NX];
  static dcomplex pad3[3];
  static int indexmap[NZ][NY][NX];

  int iter;
  int nthreads = 1;
  double total_time, mflops;
  boolean verified;
  char cls;

  /*--------------------------------------------------------------------
  c Run the entire problem once to make sure all data is touched.
  c This reduces variable startup costs, which is important for such a
  c short benchmark. The other NPB 2 implementations are similar.
  c-------------------------------------------------------------------*/
  for (i = 0; i < T_MAX; i++) {
    timer_clear(i);
  }
  setup();

  compute_indexmap(indexmap, dims[2]);

  compute_initial_conditions(u1, dims[0]);
  fft_init(dims[0][0]);

  fft(1, u1, u0);

  /*--------------------------------------------------------------------
  c Start over from the beginning. Note that all operations must
  c be timed, in contrast to other benchmarks.
  c-------------------------------------------------------------------*/
  for (i = 0; i < T_MAX; i++) {
    timer_clear(i);
  }

  timer_start(T_TOTAL);
  if (TIMERS_ENABLED == TRUE)
    timer_start(T_SETUP);

  compute_indexmap(indexmap, dims[2]);

  compute_initial_conditions(u1, dims[0]);
  fft_init(dims[0][0]);

  if (TIMERS_ENABLED == TRUE) {
    timer_stop(T_SETUP);
  }
  if (TIMERS_ENABLED == TRUE) {
    timer_start(T_FFT);
  }

  fft(1, u1, u0);
  if (TIMERS_ENABLED == TRUE) {
    timer_stop(T_FFT);
  }

  for (iter = 1; iter <= niter; iter++) {
    if (TIMERS_ENABLED == TRUE) {
      timer_start(T_EVOLVE);
    }

    evolve(u0, u1, iter, indexmap, dims[0]);

    if (TIMERS_ENABLED == TRUE) {
      timer_stop(T_EVOLVE);
    }
    if (TIMERS_ENABLED == TRUE) {
      timer_start(T_FFT);
    }

    fft(-1, u1, u2);

    if (TIMERS_ENABLED == TRUE) {
      timer_stop(T_FFT);
    }
    if (TIMERS_ENABLED == TRUE) {
      timer_start(T_CHECKSUM);
    }
    checksum(iter, u2, dims[0]);

    if (TIMERS_ENABLED == TRUE) {
      timer_stop(T_CHECKSUM);
    }
  }

  verify(NX, NY, NZ, niter, &verified, &cls);
  timer_stop(T_TOTAL);
  total_time = timer_read(T_TOTAL);

  if (total_time != 0.0) {
    mflops = 1.0e-6 * (double)(NTOTAL) *
             (14.8157 + 7.19641 * log((double)(NTOTAL)) +
              (5.23518 + 7.21113 * log((double)(NTOTAL))) * niter) /
             total_time;
  } else {
    mflops = 0.0;
  }
  c_print_results("FT", cls, NX, NY, NZ, niter, nthreads, total_time, mflops,
                  "          floating point", verified, NPBVERSION, COMPILETIME,
                  CS1, CS2, CS3, CS4, CS5, CS6, CS7);
  if (TIMERS_ENABLED == TRUE)
    print_timers();
}

/*--------------------------------------------------------------------
c-------------------------------------------------------------------*/

void evolve(dcomplex u0[NZ][NY][NX], dcomplex u1[NZ][NY][NX], int t,
                   int indexmap[NZ][NY][NX], int d[3]) {

  /*--------------------------------------------------------------------
  c-------------------------------------------------------------------*/

  /*--------------------------------------------------------------------
  c evolve u0 -> u1 (t time steps) in fourier space
  c-------------------------------------------------------------------*/

  auto irange = view::ints(0, d[0]);
  auto jrange = view::ints(0, d[1]);
  auto krange = view::ints(0, d[2]);
  /*std::vector<int> irange(d[0]);
  std::iota(irange.begin(),irange.end(),0);
  std::vector<int> jrange(d[1]);
  std::iota(jrange.begin(),jrange.end(),0);
  std::vector<int> krange(d[2]);
  std::iota(krange.begin(),krange.end(),0);*/
  NS::for_each(PARALLEL,irange.begin(), irange.end(), [&](auto i) -> void {
    NS::for_each(PARALLEL,jrange.begin(), jrange.end(), [&](auto j) -> void {
      NS::for_each(PARALLELUNSEQ,krange.begin(), krange.end(), [&](auto k) -> void {
        crmul(u1[k][j][i], u0[k][j][i], ex[t * indexmap[k][j][i]]);
      });
    });
  });
}

/*--------------------------------------------------------------------
c-------------------------------------------------------------------*/

void compute_initial_conditions(dcomplex u0[NZ][NY][NX], int d[3]) {

  /*--------------------------------------------------------------------
  c-------------------------------------------------------------------*/

  /*--------------------------------------------------------------------
  c Fill in array u0 with initial conditions from
  c random number generator
  c-------------------------------------------------------------------*/

  int k;
  double x0, start, an, dummy;
  static double tmp[NX * 2 * MAXDIM + 1];
  int i, j, t;

  start = SEED;
  /*--------------------------------------------------------------------
  c Jump to the starting element for our first plane.
  c-------------------------------------------------------------------*/
  ipow46(A, (zstart[0] - 1) * 2 * NX * NY + (ystart[0] - 1) * 2 * NX, &an);
  dummy = randlc(&start, an);
  ipow46(A, 2 * NX * NY, &an);

  /*--------------------------------------------------------------------
  c Go through by z planes filling in one square at a time.
  c-------------------------------------------------------------------*/
  for (k = 0; k < dims[0][2]; k++) {
    x0 = start;
    vranlc(2 * NX * dims[0][1], &x0, A, tmp);

    t = 1;
    for (j = 0; j < dims[0][1]; j++)
      for (i = 0; i < NX; i++) {
        u0[k][j][i].real = tmp[t++];
        u0[k][j][i].imag = tmp[t++];
      }

    if (k != dims[0][2])
      dummy = randlc(&start, an);
  }
}

/*--------------------------------------------------------------------
c-------------------------------------------------------------------*/

void ipow46(double a, int exponent, double *result) {

  /*--------------------------------------------------------------------
  c-------------------------------------------------------------------*/

  /*--------------------------------------------------------------------
  c compute a^exponent mod 2^46
  c-------------------------------------------------------------------*/

  double dummy, q, r;
  int n, n2;

  /*--------------------------------------------------------------------
  c Use
  c   a^n = a^(n/2)*a^(n/2) if n even else
  c   a^n = a*a^(n-1)       if n odd
  c-------------------------------------------------------------------*/
  *result = 1;
  if (exponent == 0)
    return;
  q = a;
  r = 1;
  n = exponent;

  while (n > 1) {
    n2 = n / 2;
    if (n2 * 2 == n) {
      dummy = randlc(&q, q);
      n = n2;
    } else {
      dummy = randlc(&r, q);
      n = n - 1;
    }
  }
  dummy = randlc(&r, q);
  *result = r;
}

/*--------------------------------------------------------------------
c-------------------------------------------------------------------*/

void setup(void) {

  /*--------------------------------------------------------------------
  c-------------------------------------------------------------------*/

  int ierr, i, j, fstatus;

  printf("\n\n NAS Parallel Benchmarks 3.0 structured OpenMP C version"
         " - FT Benchmark\n\n");

  niter = NITER_DEFAULT;

  printf(" Size                : %3dx%3dx%3d\n", NX, NY, NZ);
  printf(" Iterations          :     %7d\n", niter);

  /* 1004 format(' Number of processes :     ', i7)
   1005 format(' Processor array     :     ', i3, 'x', i3)
   1006 format(' WARNING: compiled for ', i5, ' processes. ',
       >       ' Will not verify. ')*/

  /*for (i = 0;i < 3 ; i++) {
      dims[i][0] = NX;
      dims[i][1] = NY;
      dims[i][2] = NZ;
}*/
  auto irange = view::ints(0, 3);
  NS::for_each(PARALLELUNSEQ,irange.begin(), irange.end(), [&](int i) -> void {
    dims[i][0] = NX;
    dims[i][1] = NY;
    dims[i][2] = NZ;
  });

  /*for (i = 0; i < 3; i++) {
      xstart[i] = 1;
      xend[i]   = NX;
      ystart[i] = 1;
      yend[i]   = NY;
      zstart[i] = 1;
      zend[i]   = NZ;
  }*/

  NS::for_each(PARALLELUNSEQ,irange.begin(), irange.end(), [&](int i) -> void {
    xstart[i] = 1;
    xend[i] = NX;
    ystart[i] = 1;
    yend[i] = NY;
    zstart[i] = 1;
    zend[i] = NZ;
  });

  /*--------------------------------------------------------------------
  c Set up info for blocking of ffts and transposes.  This improves
  c performance on cache-based systems. Blocking involves
  c working on a chunk of the problem at a time, taking chunks
  c along the first, second, or third dimension.
  c
  c - In cffts1 blocking is on 2nd dimension (with fft on 1st dim)
  c - In cffts2/3 blocking is on 1st dimension (with fft on 2nd and 3rd dims)

  c Since 1st dim is always in processor, we'll assume it's long enough
  c (default blocking factor is 16 so min size for 1st dim is 16)
  c The only case we have to worry about is cffts1 in a 2d decomposition.
  c so the blocking factor should not be larger than the 2nd dimension.
  c-------------------------------------------------------------------*/

  fftblock = FFTBLOCK_DEFAULT;
  fftblockpad = FFTBLOCKPAD_DEFAULT;

  if (fftblock != FFTBLOCK_DEFAULT)
    fftblockpad = fftblock + 3;
}

/*--------------------------------------------------------------------
c-------------------------------------------------------------------*/

void compute_indexmap(int indexmap[NZ][NY][NX], int d[3]) {

  /*--------------------------------------------------------------------
  c-------------------------------------------------------------------*/

  /*--------------------------------------------------------------------
  c compute function from local (i,j,k) to ibar^2+jbar^2+kbar^2
  c for time evolution exponent.
  c-------------------------------------------------------------------*/

  // int i, j, k, ii, ii2, jj, ij2, kk;
  // double ap;

  /*--------------------------------------------------------------------
  c basically we want to convert the fortran indices
  c   1 2 3 4 5 6 7 8
  c to
  c   0 1 2 3 -4 -3 -2 -1
  c The following magic formula does the trick:
  c mod(i-1+n/2, n) - n/2
  c-------------------------------------------------------------------*/

  /*#pragma omp parallel for default(shared) private(i, j, k, ii, ii2, jj, ij2,
    kk) for (i = 0; i < dims[2][0]; i++) {*/
  auto irange = view::ints(0, dims[2][0]);
  NS::for_each(PARALLEL,irange.begin(), irange.end(), [&](auto i) -> void {
    int ii = (i + 1 + xstart[2] - 2 + NX / 2) % NX - NX / 2;
    int ii2 = ii * ii;
    auto jrange = view::ints(0, dims[2][1]);
    NS::for_each(PARALLEL,jrange.begin(), jrange.end(), [&](auto j) -> void {
      // for (j = 0; j < dims[2][1]; j++) {
      int jj = (j + 1 + ystart[2] - 2 + NY / 2) % NY - NY / 2;
      int ij2 = jj * jj + ii2;
      auto krange = view::ints(0, dims[2][2]);
      // for (k = 0; k < dims[2][2]; k++) {
      NS::for_each(PARALLELUNSEQ,krange.begin(), krange.end(), [&](auto k) -> void {
        int kk = (k + 1 + zstart[2] - 2 + NZ / 2) % NZ - NZ / 2;
        indexmap[k][j][i] = kk * kk + ij2;
      });
    });
  });

  /*--------------------------------------------------------------------
  c compute array of exponentials for time evolution.
  c-------------------------------------------------------------------*/
  double ap = -4.0 * ALPHA * PI * PI;

  ex[0] = 1.0;
  ex[1] = exp(ap);

  /*for (i = 2; i <= EXPMAX; i++) {
    ex[i] = ex[i - 1] * ex[1];
  }*/
  irange = view::ints(2, EXPMAX + 1);
  NS::for_each(PARALLEL,irange.begin(), irange.end(),
                [&](auto i) -> void { ex[i] = ex[i - 1] * ex[1]; });
}

/*--------------------------------------------------------------------
c-------------------------------------------------------------------*/

void print_timers(void) {

  /*--------------------------------------------------------------------
  c-------------------------------------------------------------------*/

  int i;
  char *tstrings[] = {"          total ", "          setup ",
                      "            fft ", "         evolve ",
                      "       checksum ", "         fftlow ",
                      "        fftcopy "};

  for (i = 0; i < T_MAX; i++) {
    if (timer_read(i) != 0.0) {
      printf("timer %2d(%16s( :%10.6f\n", i, tstrings[i], timer_read(i));
    }
  }
}

/*--------------------------------------------------------------------
c-------------------------------------------------------------------*/

void fft(int dir, dcomplex x1[NZ][NY][NX], dcomplex x2[NZ][NY][NX]) {

  /*--------------------------------------------------------------------
  c-------------------------------------------------------------------*/

  dcomplex y0[NX][FFTBLOCKPAD];
  dcomplex y1[NX][FFTBLOCKPAD];

  /*--------------------------------------------------------------------
  c note: args x1, x2 must be different arrays
  c note: args for cfftsx are (direction, layout, xin, xout, scratch)
  c       xin/xout may be the same and it can be somewhat faster
  c       if they are
  c-------------------------------------------------------------------*/

  if (dir == 1) {
    cffts1(1, dims[0], x1, x1, y0, y1); /* x1 -> x1 */
    cffts2(1, dims[1], x1, x1, y0, y1); /* x1 -> x1 */
    cffts3(1, dims[2], x1, x2, y0, y1); /* x1 -> x2 */
  } else {
    cffts3(-1, dims[2], x1, x1, y0, y1); /* x1 -> x1 */
    cffts2(-1, dims[1], x1, x1, y0, y1); /* x1 -> x1 */
    cffts1(-1, dims[0], x1, x2, y0, y1); /* x1 -> x2 */
  }
}

/*--------------------------------------------------------------------
c-------------------------------------------------------------------*/

void cffts1(int is, int d[3], dcomplex x[NZ][NY][NX],
                   dcomplex xout[NZ][NY][NX], dcomplex y0[NX][FFTBLOCKPAD],
                   dcomplex y1[NX][FFTBLOCKPAD]) {

  /*--------------------------------------------------------------------
  c-------------------------------------------------------------------*/

  int logd[3];
  // int i, j, k, jj;
  auto irange = view::ints(0, 3);
  NS::for_each(PARALLEL,irange.begin(), irange.end(),
                [&](auto i) -> void { logd[i] = ilog2(d[i]); });
  irange = view::ints(0, d[0]);
  auto jrange = view::ints(0, fftblock);
  auto krange = view::ints(0, d[2]);
  auto jjrange = view::ints(0, ((d[1] - fftblock) / fftblock) + 1);

  NS::for_each(PARALLEL,krange.begin(), krange.end(), [&](auto k) -> void {
    NS::for_each(PARALLEL,jjrange.begin(), jjrange.end(), [&](auto jj) -> void {
      jj = jj * fftblock;
      NS::for_each(PARALLEL,jrange.begin(), jrange.end(), [&](auto j) -> void {
        NS::for_each(PARALLELUNSEQ,irange.begin(), irange.end(), [&](auto i) -> void {
          y0[i][j].real = x[k][j + jj][i].real;

          y0[i][j].imag = x[k][j + jj][i].imag;
        });
      });
      cfftz(is, logd[0], d[0], y0, y1);
      NS::for_each(PARALLEL,jrange.begin(), jrange.end(), [&](auto j) -> void {
        NS::for_each(PARALLELUNSEQ,irange.begin(), irange.end(), [&](auto i) -> void {
          xout[k][j + jj][i].real = y0[i][j].real;
          xout[k][j + jj][i].imag = y0[i][j].imag;
        });
      });
    });
  });
  //}*/
}

/*--------------------------------------------------------------------
c-------------------------------------------------------------------*/

void cffts2(int is, int d[3], dcomplex x[NZ][NY][NX],
                   dcomplex xout[NZ][NY][NX], dcomplex y0[NX][FFTBLOCKPAD],
                   dcomplex y1[NX][FFTBLOCKPAD]) {

  /*--------------------------------------------------------------------
  c-------------------------------------------------------------------*/

  int logd[3];
  int i, j, k, ii;

  auto irange = view::ints(0, 3);
  NS::for_each(PARALLEL,irange.begin(), irange.end(),
                [&](auto i) -> void { logd[i] = ilog2(d[i]); });
		auto krange = view::ints(0, d[2]);
		auto iirange = view::ints(0, ((d[0] - fftblock) / fftblock) + 1);
			NS::for_each(PARALLEL,krange.begin(), krange.end(), [&](auto k) -> void {
		    NS::for_each(PARALLEL,iirange.begin(), iirange.end(), [&](auto ii) -> void {
        irange = view::ints(0, d[1]);
        auto jrange = view::ints(0, fftblock);
        NS::for_each(PARALLEL,jrange.begin(), jrange.end(), [&](auto j) -> void {
          NS::for_each(PARALLELUNSEQ,irange.begin(), irange.end(), [&](auto i) -> void {
            y0[j][i].real = x[k][j][i + ii].real;
            y0[j][i].imag = x[k][j][i + ii].imag;
          });
        });
        cfftz(is, logd[1], d[1], y0, y1);
        irange = view::ints(0, fftblock);
        jrange = view::ints(0, d[1]);
        NS::for_each(PARALLEL,jrange.begin(), jrange.end(), [&](auto j) -> void {
          NS::for_each(PARALLELUNSEQ,irange.begin(), irange.end(), [&](auto i) -> void {
            xout[k][j][i + ii].real = y0[j][i].real;
            xout[k][j][i + ii].imag = y0[j][i].imag;
          });
        });
      });
    });
}
/*--------------------------------------------------------------------
c-------------------------------------------------------------------*/

void cffts3(int is, int d[3], dcomplex x[NZ][NY][NX],
                   dcomplex xout[NZ][NY][NX], dcomplex y0[NX][FFTBLOCKPAD],
                   dcomplex y1[NX][FFTBLOCKPAD]) {

  /*--------------------------------------------------------------------
  c-------------------------------------------------------------------*/

  int logd[3];
  int i, j, k, ii;
  auto irange = view::ints(0, 3);
  NS::for_each(PARALLEL,irange.begin(), irange.end(),
                [&](auto i) -> void { logd[i] = ilog2(d[i]); });
  auto jrange = view::ints(0, d[1]);
  NS::for_each(PARALLEL,jrange.begin(), jrange.end(), [&](auto j) -> void {
    int ii = 0;
		auto iirange = view::ints(0, ((d[0] - fftblock) / fftblock) + 1);
		NS::for_each(PARALLEL,iirange.begin(), iirange.end(), [&](auto ii) -> void {
      auto krange = view::ints(0, d[2]);
      NS::for_each(PARALLEL,krange.begin(), krange.end(), [&](auto k) -> void {
        auto irange2 = view::ints(0, fftblock);
        NS::for_each(PARALLELUNSEQ,irange2.begin(), irange2.end(), [&](auto i) -> void {
          y0[k][i].real = x[k][j][i + ii].real;
          y0[k][i].imag = x[k][j][i + ii].imag;
        });
      });
      cfftz(is, logd[2], d[2], y0, y1);
      krange = view::ints(0, d[2]);
      irange = view::ints(0, fftblock);
      NS::for_each(PARALLEL,krange.begin(), krange.end(), [&](auto k) -> void {
        NS::for_each(PARALLELUNSEQ,irange.begin(), irange.end(), [&](auto i) -> void {
          xout[k][j][i + ii].real = y0[k][i].real;
          xout[k][j][i + ii].imag = y0[k][i].imag;
        });
      });
    });
  });
}

/*--------------------------------------------------------------------
c-------------------------------------------------------------------*/

void fft_init(int n) {

  /*--------------------------------------------------------------------
  c-------------------------------------------------------------------*/

  /*--------------------------------------------------------------------
  c compute the roots-of-unity array that will be used for subsequent FFTs.
  c-------------------------------------------------------------------*/

  int m, nu, ku, i, j, ln;
  double t, ti;

  /*--------------------------------------------------------------------
  c   Initialize the U array with sines and cosines in a manner that permits
  c   stride one access at each FFT iteration.
  c-------------------------------------------------------------------*/
  nu = n;
  m = ilog2(n);
  u[0].real = (double)m;
  u[0].imag = 0.0;
  ku = 1;
  ln = 1;

  for (j = 1; j <= m; j++) {
    t = PI / ln;

    for (i = 0; i <= ln - 1; i++) {
      ti = i * t;
      u[i + ku].real = cos(ti);
      u[i + ku].imag = sin(ti);
    }

    ku = ku + ln;
    ln = 2 * ln;
  }
}

/*--------------------------------------------------------------------
c-------------------------------------------------------------------*/

void cfftz(int is, int m, int n, dcomplex x[NX][FFTBLOCKPAD],
                  dcomplex y[NX][FFTBLOCKPAD]) {

  /*--------------------------------------------------------------------
  c-------------------------------------------------------------------*/

  /*--------------------------------------------------------------------
  c   Computes NY N-point complex-to-complex FFTs of X using an algorithm due
  c   to Swarztrauber.  X is both the input and the output array, while Y is a
  c   scratch array.  It is assumed that N = 2^M.  Before calling CFFTZ to
  c   perform FFTs, the array U must be initialized by calling CFFTZ with IS
  c   set to 0 and M set to MX, where MX is the maximum value of M for any
  c   subsequent call.
  c-------------------------------------------------------------------*/

  int i, j, l, mx;

  /*--------------------------------------------------------------------
  c   Check if input parameters are invalid.
  c-------------------------------------------------------------------*/
  mx = (int)(u[0].real);
  if ((is != 1 && is != -1) || m < 1 || m > mx) {
    printf("CFFTZ: Either U has not been initialized, or else\n"
           "one of the input parameters is invalid%5d%5d%5d\n",
           is, m, mx);
    exit(1);
  }

  /*--------------------------------------------------------------------
  c   Perform one variant of the Stockham FFT.
  c-------------------------------------------------------------------*/
  for (l = 1; l <= m; l += 2) {
    fftz2(is, l, m, n, fftblock, fftblockpad, u, x, y);
    if (l == m)
      break;
    fftz2(is, l + 1, m, n, fftblock, fftblockpad, u, y, x);
  }

  /*--------------------------------------------------------------------
  c   Copy Y to X.
  c-------------------------------------------------------------------*/
  if (m % 2 == 1) {
    for (j = 0; j < n; j++) {
      for (i = 0; i < fftblock; i++) {
        x[j][i].real = y[j][i].real;
        x[j][i].imag = y[j][i].imag;
      }
    }
  }
}

/*--------------------------------------------------------------------
c-------------------------------------------------------------------*/

void fftz2(int is, int l, int m, int n, int ny, int ny1, dcomplex u[NX],
                  dcomplex x[NX][FFTBLOCKPAD], dcomplex y[NX][FFTBLOCKPAD]) {

  /*--------------------------------------------------------------------
  c-------------------------------------------------------------------*/

  /*--------------------------------------------------------------------
  c   Performs the L-th iteration of the second variant of the Stockham FFT.
  c-------------------------------------------------------------------*/

  int k, n1, li, lj, lk, ku, i, j, i11, i12, i21, i22;
  dcomplex u1, x11, x21;

  /*--------------------------------------------------------------------
  c   Set initial parameters.
  c-------------------------------------------------------------------*/

  n1 = n / 2;
  if (l - 1 == 0) {
    lk = 1;
  } else {
    lk = 2 << ((l - 1) - 1);
  }
  if (m - l == 0) {
    li = 1;
  } else {
    li = 2 << ((m - l) - 1);
  }
  lj = 2 * lk;
  ku = li;

  for (i = 0; i < li; i++) {

    i11 = i * lk;
    i12 = i11 + n1;
    i21 = i * lj;
    i22 = i21 + lk;
    if (is >= 1) {
      u1.real = u[ku + i].real;
      u1.imag = u[ku + i].imag;
    } else {
      u1.real = u[ku + i].real;
      u1.imag = -u[ku + i].imag;
    }

    /*--------------------------------------------------------------------
    c   This loop is vectorizable.
    c-------------------------------------------------------------------*/
    //for (k = 0; k < lk; k++) {
    auto krange = view::ints(0, lk);
    auto jrange = view::ints(0, ny);
    NS::for_each(PARALLEL,krange.begin(), krange.end(), [&](auto k) -> void {
      //for (j = 0; j < ny; j++) {
    NS::for_each(PARALLELUNSEQ,jrange.begin(), jrange.end(), [&](auto j) -> void {
        double x11real, x11imag;
        double x21real, x21imag;
        x11real = x[i11 + k][j].real;
        x11imag = x[i11 + k][j].imag;
        x21real = x[i12 + k][j].real;
        x21imag = x[i12 + k][j].imag;
        y[i21 + k][j].real = x11real + x21real;
        y[i21 + k][j].imag = x11imag + x21imag;
        y[i22 + k][j].real =
            u1.real * (x11real - x21real) - u1.imag * (x11imag - x21imag);
        y[i22 + k][j].imag =
            u1.real * (x11imag - x21imag) + u1.imag * (x11real - x21real);
      });
    });
  }
}

/*--------------------------------------------------------------------
c-------------------------------------------------------------------*/

static int ilog2(int n) {

  /*--------------------------------------------------------------------
  c-------------------------------------------------------------------*/

  int nn, lg;

  if (n == 1) {
    return 0;
  }
  lg = 1;
  nn = 2;
  while (nn < n) {
    nn = nn << 1;
    lg++;
  }

  return lg;
}

/*--------------------------------------------------------------------
c-------------------------------------------------------------------*/

void checksum(int i, dcomplex u1[NZ][NY][NX], int d[3]) {


    /*--------------------------------------------------------------------
    c-------------------------------------------------------------------*/

    int j, q, r, s, ierr;
    dcomplex chk, allchk;

    chk.real = 0.0;
    chk.imag = 0.0;
  auto jrange=view::ints(1,1025);
    NS::for_each(PARALLEL,jrange.begin(), jrange.end(), [&](auto j) -> void {
      q = j % NX + 1;
      if (q >= xstart[0] && q <= xend[0]) {
        r = (3 * j) % NY + 1;
        if (r >= ystart[0] && r <= yend[0]) {
          s = (5 * j) % NZ + 1;
          if (s >= zstart[0] && s <= zend[0]) {
            cadd(chk, chk, u1[s - zstart[0]][r - ystart[0]][q - xstart[0]]);
          }
        }
      }
    });
      sums[i].real += chk.real;
      sums[i].imag += chk.imag;
    {
      sums[i].real = sums[i].real / (double)(NTOTAL);
      sums[i].imag = sums[i].imag / (double)(NTOTAL);

      printf("T = %5d     Checksum = %22.12e %22.12e\n", i, sums[i].real,
             sums[i].imag);
    }
}

/*--------------------------------------------------------------------
c-------------------------------------------------------------------*/

void verify(int d1, int d2, int d3, int nt, boolean *verified,
                   char *cls) {

  /*--------------------------------------------------------------------
  c-------------------------------------------------------------------*/

  int ierr, size, i;
  double err, epsilon;

  /*--------------------------------------------------------------------
  c   Sample size reference checksums
  c-------------------------------------------------------------------*/

  /*--------------------------------------------------------------------
  c   Class S size reference checksums
  c-------------------------------------------------------------------*/
  double vdata_real_s[6 + 1] = {0.0,
                                5.546087004964e+02,
                                5.546385409189e+02,
                                5.546148406171e+02,
                                5.545423607415e+02,
                                5.544255039624e+02,
                                5.542683411902e+02};
  double vdata_imag_s[6 + 1] = {0.0,
                                4.845363331978e+02,
                                4.865304269511e+02,
                                4.883910722336e+02,
                                4.901273169046e+02,
                                4.917475857993e+02,
                                4.932597244941e+02};
  /*--------------------------------------------------------------------
  c   Class W size reference checksums
  c-------------------------------------------------------------------*/
  double vdata_real_w[6 + 1] = {0.0,
                                5.673612178944e+02,
                                5.631436885271e+02,
                                5.594024089970e+02,
                                5.560698047020e+02,
                                5.530898991250e+02,
                                5.504159734538e+02};
  double vdata_imag_w[6 + 1] = {0.0,
                                5.293246849175e+02,
                                5.282149986629e+02,
                                5.270996558037e+02,
                                5.260027904925e+02,
                                5.249400845633e+02,
                                5.239212247086e+02};
  /*--------------------------------------------------------------------
  c   Class A size reference checksums
  c-------------------------------------------------------------------*/
  double vdata_real_a[6 + 1] = {0.0,
                                5.046735008193e+02,
                                5.059412319734e+02,
                                5.069376896287e+02,
                                5.077892868474e+02,
                                5.085233095391e+02,
                                5.091487099959e+02};
  double vdata_imag_a[6 + 1] = {0.0,
                                5.114047905510e+02,
                                5.098809666433e+02,
                                5.098144042213e+02,
                                5.101336130759e+02,
                                5.104914655194e+02,
                                5.107917842803e+02};
  /*--------------------------------------------------------------------
  c   Class B size reference checksums
  c-------------------------------------------------------------------*/
  double vdata_real_b[20 + 1] = {0.0,
                                 5.177643571579e+02,
                                 5.154521291263e+02,
                                 5.146409228649e+02,
                                 5.142378756213e+02,
                                 5.139626667737e+02,
                                 5.137423460082e+02,
                                 5.135547056878e+02,
                                 5.133910925466e+02,
                                 5.132470705390e+02,
                                 5.131197729984e+02,
                                 5.130070319283e+02,
                                 5.129070537032e+02,
                                 5.128182883502e+02,
                                 5.127393733383e+02,
                                 5.126691062020e+02,
                                 5.126064276004e+02,
                                 5.125504076570e+02,
                                 5.125002331720e+02,
                                 5.124551951846e+02,
                                 5.124146770029e+02};
  double vdata_imag_b[20 + 1] = {0.0,
                                 5.077803458597e+02,
                                 5.088249431599e+02,
                                 5.096208912659e+02,
                                 5.101023387619e+02,
                                 5.103976610617e+02,
                                 5.105948019802e+02,
                                 5.107404165783e+02,
                                 5.108576573661e+02,
                                 5.109577278523e+02,
                                 5.110460304483e+02,
                                 5.111252433800e+02,
                                 5.111968077718e+02,
                                 5.112616233064e+02,
                                 5.113203605551e+02,
                                 5.113735928093e+02,
                                 5.114218460548e+02,
                                 5.114656139760e+02,
                                 5.115053595966e+02,
                                 5.115415130407e+02,
                                 5.115744692211e+02};
  /*--------------------------------------------------------------------
  c   Class C size reference checksums
  c-------------------------------------------------------------------*/
  double vdata_real_c[20 + 1] = {0.0,
                                 5.195078707457e+02,
                                 5.155422171134e+02,
                                 5.144678022222e+02,
                                 5.140150594328e+02,
                                 5.137550426810e+02,
                                 5.135811056728e+02,
                                 5.134569343165e+02,
                                 5.133651975661e+02,
                                 5.132955192805e+02,
                                 5.132410471738e+02,
                                 5.131971141679e+02,
                                 5.131605205716e+02,
                                 5.131290734194e+02,
                                 5.131012720314e+02,
                                 5.130760908195e+02,
                                 5.130528295923e+02,
                                 5.130310107773e+02,
                                 5.130103090133e+02,
                                 5.129905029333e+02,
                                 5.129714421109e+02};
  double vdata_imag_c[20 + 1] = {0.0,
                                 5.149019699238e+02,
                                 5.127578201997e+02,
                                 5.122251847514e+02,
                                 5.121090289018e+02,
                                 5.121143685824e+02,
                                 5.121496764568e+02,
                                 5.121870921893e+02,
                                 5.122193250322e+02,
                                 5.122454735794e+02,
                                 5.122663649603e+02,
                                 5.122830879827e+02,
                                 5.122965869718e+02,
                                 5.123075927445e+02,
                                 5.123166486553e+02,
                                 5.123241541685e+02,
                                 5.123304037599e+02,
                                 5.123356167976e+02,
                                 5.123399592211e+02,
                                 5.123435588985e+02,
                                 5.123465164008e+02};

  epsilon = 1.0e-12;
  *verified = TRUE;
  *cls = 'U';

  if (d1 == 64 && d2 == 64 && d3 == 64 && nt == 6) {
    *cls = 'S';
    for (i = 1; i <= nt; i++) {
      err = (get_real(sums[i]) - vdata_real_s[i]) / vdata_real_s[i];
      if (fabs(err) > epsilon) {
        *verified = FALSE;
        break;
      }
      err = (get_imag(sums[i]) - vdata_imag_s[i]) / vdata_imag_s[i];
      if (fabs(err) > epsilon) {
        *verified = FALSE;
        break;
      }
    }
  } else if (d1 == 128 && d2 == 128 && d3 == 32 && nt == 6) {
    *cls = 'W';
    for (i = 1; i <= nt; i++) {
      err = (get_real(sums[i]) - vdata_real_w[i]) / vdata_real_w[i];
      if (fabs(err) > epsilon) {
        *verified = FALSE;
        break;
      }
      err = (get_imag(sums[i]) - vdata_imag_w[i]) / vdata_imag_w[i];
      if (fabs(err) > epsilon) {
        *verified = FALSE;
        break;
      }
    }
  } else if (d1 == 256 && d2 == 256 && d3 == 128 && nt == 6) {
    *cls = 'A';
    for (i = 1; i <= nt; i++) {
      err = (get_real(sums[i]) - vdata_real_a[i]) / vdata_real_a[i];
      if (fabs(err) > epsilon) {
        *verified = FALSE;
        break;
      }
      err = (get_imag(sums[i]) - vdata_imag_a[i]) / vdata_imag_a[i];
      if (fabs(err) > epsilon) {
        *verified = FALSE;
        break;
      }
    }
  } else if (d1 == 512 && d2 == 256 && d3 == 256 && nt == 20) {
    *cls = 'B';
    for (i = 1; i <= nt; i++) {
      err = (get_real(sums[i]) - vdata_real_b[i]) / vdata_real_b[i];
      if (fabs(err) > epsilon) {
        *verified = FALSE;
        break;
      }
      err = (get_imag(sums[i]) - vdata_imag_b[i]) / vdata_imag_b[i];
      if (fabs(err) > epsilon) {
        *verified = FALSE;
        break;
      }
    }
  } else if (d1 == 512 && d2 == 512 && d3 == 512 && nt == 20) {
    *cls = 'C';
    for (i = 1; i <= nt; i++) {
      err = (get_real(sums[i]) - vdata_real_c[i]) / vdata_real_c[i];
      if (fabs(err) > epsilon) {
        *verified = FALSE;
        break;
      }
      err = (get_imag(sums[i]) - vdata_imag_c[i]) / vdata_imag_c[i];
      if (fabs(err) > epsilon) {
        *verified = FALSE;
        break;
      }
    }
  }

  if (*cls != 'U') {
    printf("Result verification successful\n");
  } else {
    printf("Result verification failed\n");
  }
  printf("class = %1c\n", *cls);
}
