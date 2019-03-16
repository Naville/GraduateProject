/*--------------------------------------------------------------------

  NAS Parallel Benchmarks 3.0 structured OpenMP C versions - CG

  This benchmark is an OpenMP C version of the NPB CG code.

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

  Authors: M. Yarrow
           C. Kuszmaul

  OpenMP C version: S. Satoh

  3.0 structure translation: F. Conti

--------------------------------------------------------------------*/

/*
c---------------------------------------------------------------------
c  Note: please observe that in the routine conj_grad three
c  implementations of the sparse matrix-vector multiply have
c  been supplied.  The default matrix-vector multiply is not
c  loop unrolled.  The alternate implementations are unrolled
c  to a depth of 2 and unrolled to a depth of 8.  Please
c  experiment with these to find the fastest for your particular
c  architecture.  If reporting timing results, any of these three may
c  be used without penalty.
c---------------------------------------------------------------------
*/
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
#include "npb-C.h"
#include <iostream>
#include <type_traits>
using namespace ranges;
#include "npbparams.h"

#define NZ NA *(NONZER + 1) * (NONZER + 1) + NA *(NONZER + 2)

/* global variables */

/* common /partit_size/ */
static int naa;
static int nzz;
static int firstrow;
static int lastrow;
static int firstcol;
static int lastcol;

/* common /main_int_mem/ */
static int colidx[NZ + 1];     /* colidx[1:NZ] */
static int rowstr[NA + 1 + 1]; /* rowstr[1:NA+1] */
static int iv[2 * NA + 1 + 1]; /* iv[1:2*NA+1] */
static int arow[NZ + 1];       /* arow[1:NZ] */
static int acol[NZ + 1];       /* acol[1:NZ] */

/* common /main_flt_mem/ */
static double v[NA + 1 + 1]; /* v[1:NA+1] */
static double aelt[NZ + 1];  /* aelt[1:NZ] */
static double a[NZ + 1];     /* a[1:NZ] */
static double x[NA + 2 + 1]; /* x[1:NA+2] */
static double z[NA + 2 + 1]; /* z[1:NA+2] */
static double p[NA + 2 + 1]; /* p[1:NA+2] */
static double q[NA + 2 + 1]; /* q[1:NA+2] */
static double r[NA + 2 + 1]; /* r[1:NA+2] */
// static double w[NA+2+1];	/* w[1:NA+2] */

/* common /urando/ */
static double amult;
static double tran;

/* function declarations */
void conj_grad(int colidx[], int rowstr[], double x[], double z[],
                      double a[], double p[], double q[], double r[],
                      // double w[],
                      double *rnorm);
void makea(int n, int nz, double a[], int colidx[], int rowstr[],
                  int nonzer, int firstrow, int lastrow, int firstcol,
                  int lastcol, double rcond, int arow[], int acol[],
                  double aelt[], double v[], int iv[], double shift);
void sparse(double a[], int colidx[], int rowstr[], int n, int arow[],
                   int acol[], double aelt[], int firstrow, int lastrow,
                   double x[], boolean mark[], int nzloc[], int nnza);
void sprnvc(int n, int nz, double v[], int iv[], int nzloc[],
                   int mark[]);
static int icnvrt(double x, int ipwr2);
void vecset(int n, double v[], int iv[], int *nzv, int i, double val);

/*--------------------------------------------------------------------
      program cg
--------------------------------------------------------------------*/

int main(int argc, char **argv) {
  int i, j, k, it;
  int nthreads = 1;
  double zeta;
  double rnorm;
  double norm_temp11;
  double norm_temp12;
  double t, mflops;
  char cls;
  boolean verified;
  double zeta_verify_value, epsilon;

  firstrow = 1;
  lastrow = NA;
  firstcol = 1;
  lastcol = NA;

  if (NA == 1400 && NONZER == 7 && NITER == 15 && SHIFT == 10.0) {
    cls = 'S';
    zeta_verify_value = 8.5971775078648;
  } else if (NA == 7000 && NONZER == 8 && NITER == 15 && SHIFT == 12.0) {
    cls = 'W';
    zeta_verify_value = 10.362595087124;
  } else if (NA == 14000 && NONZER == 11 && NITER == 15 && SHIFT == 20.0) {
    cls = 'A';
    zeta_verify_value = 17.130235054029;
  } else if (NA == 75000 && NONZER == 13 && NITER == 75 && SHIFT == 60.0) {
    cls = 'B';
    zeta_verify_value = 22.712745482631;
  } else if (NA == 150000 && NONZER == 15 && NITER == 75 && SHIFT == 110.0) {
    cls = 'C';
    zeta_verify_value = 28.973605592845;
  } else {
    cls = 'U';
  }

  printf("\n\n NAS Parallel Benchmarks 3.0 structured OpenMP C version"
         " - CG Benchmark\n");
  printf(" Size: %10d\n", NA);
  printf(" Iterations: %5d\n", NITER);

  naa = NA;
  nzz = NZ;

  /*--------------------------------------------------------------------
  c  Initialize random number generator
  c-------------------------------------------------------------------*/
  tran = 314159265.0;
  amult = 1220703125.0;
  zeta = randlc(&tran, amult);

  /*--------------------------------------------------------------------
  c
  c-------------------------------------------------------------------*/
  makea(naa, nzz, a, colidx, rowstr, NONZER, firstrow, lastrow, firstcol,
        lastcol, RCOND, arow, acol, aelt, v, iv, SHIFT);

/*---------------------------------------------------------------------
c  Note: as a result of the above call to makea:
c        values of j used in indexing rowstr go from 1 --> lastrow-firstrow+1
c        values of colidx which are col indexes go from firstcol --> lastcol
c        So:
c        Shift the col index vals from actual (firstcol --> lastcol )
c        to local, i.e., (1 --> lastcol-firstcol+1)
c---------------------------------------------------------------------*/
auto jrange=view::ints(0,1);
auto irange=view::ints(0,1);
    jrange=view::ints(1,lastrow - firstrow + 2);
    NS::for_each(PARALLEL,jrange.begin(), jrange.end(), [&](auto j) -> void {
      auto k2=view::ints(rowstr[j],rowstr[j + 1]);
      NS::for_each(PARALLELUNSEQ,k2.begin(), k2.end(), [&](auto k) -> void {
        colidx[k] = colidx[k] - firstcol + 1;
      });
    });

/*--------------------------------------------------------------------
c  set starting vector to (1, 1, .... 1)
c-------------------------------------------------------------------*/
  irange=view::ints(1,NA+2);
    NS::for_each(PARALLEL,irange.begin(), irange.end(), [&](auto i) -> void {
      x[i] = 1.0;
    });
  jrange=view::ints(1,lastcol - firstcol + 2);
    NS::for_each(PARALLELUNSEQ,jrange.begin(), jrange.end(), [&](auto j) -> void {
      q[j] = 0.0;
      z[j] = 0.0;
      r[j] = 0.0;
      p[j] = 0.0;
    });
  zeta = 0.0;

  /*-------------------------------------------------------------------
  c---->
  c  Do one iteration untimed to init all code and data page tables
  c---->                    (then reinit, start timing, to niter its)
  c-------------------------------------------------------------------*/

  for (it = 1; it <= 1; it++) {

    /*--------------------------------------------------------------------
    c  The call to the conjugate gradient routine:
    c-------------------------------------------------------------------*/
    conj_grad(colidx, rowstr, x, z, a, p, q, r, /* w,*/ &rnorm);

    /*--------------------------------------------------------------------
    c  zeta = shift + 1/(x.z)
    c  So, first: (x.z)
    c  Also, find norm of z
    c  So, first: (z.z)
    c-------------------------------------------------------------------*/
    norm_temp11=std::accumulate(&x[1],&x[lastcol - firstcol + 2],0.0,[=](double i,double j)->double{return i+j*j;});
    norm_temp12=std::accumulate(&z[1],&z[lastcol - firstcol + 2],0.0,[=](double i,double j)->double{return i+j*j;});
    norm_temp12 = 1.0 / sqrt(norm_temp12);

/*--------------------------------------------------------------------
c  Normalize z to obtain x
c-------------------------------------------------------------------*/
    jrange=view::ints(1,lastcol - firstcol + 2);
    NS::for_each(PARALLELUNSEQ,jrange.begin(), jrange.end(), [&](auto j) -> void {
      x[j] = norm_temp12 * z[j];
    });

  } /* end of do one iteration untimed */

/*--------------------------------------------------------------------
c  set starting vector to (1, 1, .... 1)
c-------------------------------------------------------------------*/
  irange=view::ints(1,NA + 2);
  NS::for_each(PARALLELUNSEQ,irange.begin(), irange.end(), [&](auto i) -> void {
    x[i] = 1.0;
  });
  zeta = 0.0;

  timer_clear(1);
  timer_start(1);

  /*--------------------------------------------------------------------
  c---->
  c  Main Iteration for inverse power method
  c---->
  c-------------------------------------------------------------------*/

  for (it = 1; it <= NITER; it++) {

    /*--------------------------------------------------------------------
    c  The call to the conjugate gradient routine:
    c-------------------------------------------------------------------*/
    conj_grad(colidx, rowstr, x, z, a, p, q, r /*, w*/, &rnorm);

    /*--------------------------------------------------------------------
    c  zeta = shift + 1/(x.z)
    c  So, first: (x.z)
    c  Also, find norm of z
    c  So, first: (z.z)
    c-------------------------------------------------------------------*/
    auto tmprange=view::ints(1,lastcol - firstcol + 2);
    norm_temp11=std::accumulate(tmprange.begin(),tmprange.end(),0.0,[=](double i,int j)->double{return i+x[j]*z[j];});
    norm_temp12=std::accumulate(&z[1],&z[lastcol - firstcol + 2],0.0,[=](double i,double j)->double{return i+j*j;});
    norm_temp12 = 1.0 / sqrt(norm_temp12);

    zeta = SHIFT + 1.0 / norm_temp11;

    if (it == 1) {
      printf("   iteration           ||r||                 zeta\n");
    }
    printf("    %5d       %20.14e%20.13e\n", it, rnorm, zeta);

/*--------------------------------------------------------------------
c  Normalize z to obtain x
c-------------------------------------------------------------------*/
    auto j2=view::ints(1,lastcol - firstcol + 2);
    NS::for_each(PARALLELUNSEQ,j2.begin(), j2.end(), [&](auto j) -> void {
      x[j] = norm_temp12 * z[j];
    });
  } /* end of main iter inv pow meth */

  timer_stop(1);

  /*--------------------------------------------------------------------
  c  End of timed section
  c-------------------------------------------------------------------*/

  t = timer_read(1);

  printf(" Benchmark completed\n");

  epsilon = 1.0e-10;
  if (cls != 'U') {
    if (fabs(zeta - zeta_verify_value) <= epsilon) {
      verified = TRUE;
      printf(" VERIFICATION SUCCESSFUL\n");
      printf(" Zeta is    %20.12e\n", zeta);
      printf(" Error is   %20.12e\n", zeta - zeta_verify_value);
    } else {
      verified = FALSE;
      printf(" VERIFICATION FAILED\n");
      printf(" Zeta                %20.12e\n", zeta);
      printf(" The correct zeta is %20.12e\n", zeta_verify_value);
    }
  } else {
    verified = FALSE;
    printf(" Problem size unknown\n");
    printf(" NO VERIFICATION PERFORMED\n");
  }

  if (t != 0.0) {
    mflops = (2.0 * NITER * NA) *
             (3.0 + (NONZER * (NONZER + 1)) +
              25.0 * (5.0 + (NONZER * (NONZER + 1))) + 3.0) /
             t / 1000000.0;
  } else {
    mflops = 0.0;
  }

  c_print_results("CG", cls, NA, 0, 0, NITER, nthreads, t, mflops,
                  "          floating point", verified, NPBVERSION, COMPILETIME,
                  CS1, CS2, CS3, CS4, CS5, CS6, CS7);
}

/*--------------------------------------------------------------------
c-------------------------------------------------------------------*/
void conj_grad(int colidx[], /* colidx[1:nzz] */
                      int rowstr[], /* rowstr[1:naa+1] */
                      double x[],   /* x[*] */
                      double z[],   /* z[*] */
                      double a[],   /* a[1:nzz] */
                      double p[],   /* p[*] */
                      double q[],   /* q[*] */
                      double r[],   /* r[*] */
                      // double w[],		/* w[*] */
                      double *rnorm)
/*--------------------------------------------------------------------
c-------------------------------------------------------------------*/

/*---------------------------------------------------------------------
c  Floaging point arrays here are named as in NPB1 spec discussion of
c  CG algorithm
c---------------------------------------------------------------------*/
{
  static int callcount = 0;
  double d, sum, rho, rho0, alpha, beta;
  int i, j, k;
  int cgit, cgitmax = 25;

  rho = 0.0;
  auto jrange=view::ints(1,naa+2);
  /*--------------------------------------------------------------------
  c  Initialize the CG algorithm:
  c-------------------------------------------------------------------*/

    NS::for_each(PARALLELUNSEQ,jrange.begin(), jrange.end(), [&](auto j) -> void {
        q[j] = 0.0;
        z[j] = 0.0;
        r[j] = x[j];
        p[j] = r[j];
        // w[j] = 0.0;
      });

    rho=std::accumulate(&r[1],&r[lastcol - firstcol + 2],0.0,[=](double i,double j)->double{return i+j*j;});
/*--------------------------------------------------------------------
c  rho = r.r
c  Now, obtain the norm of r: First, sum squares of r elements locally...
c-------------------------------------------------------------------*/
    /*--------------------------------------------------------------------
    c---->
    c  The conj grad iteration loop
    c---->
    c-------------------------------------------------------------------*/
  for (cgit = 1; cgit <= cgitmax; cgit++) {
    rho0 = rho;
    d = 0.0;
    rho = 0.0;
#pragma omp parallel default(shared) private(j, k, sum, alpha, beta)           \
    shared(d, rho0, rho)
    {

/*--------------------------------------------------------------------
c  q = A.p
c  The partition submatrix-vector multiply: use workspace w
c---------------------------------------------------------------------
C
C  NOTE: this version of the multiply is actually (slightly: maybe %5)
C        faster on the sp2 on 16 nodes than is the unrolled-by-2 version
C        below.   On the Cray t3d, the reverse is true, i.e., the
C        unrolled-by-two version is some 10% faster.
C        The unrolled-by-8 version below is significantly faster
C        on the Cray t3d - overall speed of code is 1.5 times faster.
*/

/* rolled version */
      jrange=view::ints(1,lastrow - firstrow + 2);
      NS::for_each(PARALLEL,jrange.begin(), jrange.end(), [&](auto j) -> void {
        sum = 0.0;
        auto k2=view::ints(rowstr[j],rowstr[j + 1]);
        NS::for_each(PARALLELUNSEQ,k2.begin(), k2.end(), [&](auto k) -> void {
          sum = sum + a[k] * p[colidx[k]];
        });
        q[j] = sum;
      });
/*--------------------------------------------------------------------
c  Obtain p.q
c-------------------------------------------------------------------*/
      d=std::accumulate(jrange.begin(),jrange.end(),0.0,[=](double i,int j)->double{return i+p[j]*q[j];});
      /*--------------------------------------------------------------------
      c  Obtain alpha = rho / (p.q)
      c-------------------------------------------------------------------*/
      alpha = rho0 / d;

      /*--------------------------------------------------------------------
      c  Save a temporary of rho
      c-------------------------------------------------------------------*/
      /*	rho0 = rho;*/

/*---------------------------------------------------------------------
c  Obtain z = z + alpha*p
c  and    r = r - alpha*q
c---------------------------------------------------------------------*/
#pragma omp for reduction(+:rho)
	for (j = 1; j <= lastcol-firstcol+1; j++) {
            z[j] = z[j] + alpha*p[j];
            r[j] = r[j] - alpha*q[j];
            rho = rho + r[j]*r[j];
}
        /*---------------------------------------------------------------------
        c  rho = r.r
        c  Now, obtain the norm of r: First, sum squares of r elements
        locally...
        c---------------------------------------------------------------------*/
#warning FIXME: Precision Round-Off issues here
      /*jrange=view::ints(1,lastcol - firstcol + 2);
        NS::for_each(PARALLEL,jrange.begin(), jrange.end(), [&](auto j) -> void {
          z[j] = z[j] + alpha * p[j];
          r[j] = r[j] - alpha * q[j];
        });
      rho = std::accumulate(jrange.begin(),jrange.end(),0.0,[=](double i,int j)->double{
        return i+r[j]*r[j];
      });*/

      /*--------------------------------------------------------------------
      c  Obtain beta:
      c-------------------------------------------------------------------*/
      //#pragma omp single
      beta = rho / rho0;

/*--------------------------------------------------------------------
c  p = r + beta*p
c-------------------------------------------------------------------*/
      jrange = view::ints(1,lastcol - firstcol + 2);
      NS::for_each(PARALLELUNSEQ,jrange.begin(), jrange.end(), [&](auto j) -> void {
        p[j] = r[j] + beta * p[j];
      });
      callcount++;
    } /* end omp parallel */
  }   /* end of do cgit=1,cgitmax */

  /*---------------------------------------------------------------------
  c  Compute residual norm explicitly:  ||r|| = ||x - A.z||
  c  First, form A.z
  c  The partition submatrix-vector multiply
  c---------------------------------------------------------------------*/
  sum = 0.0;

    jrange=view::ints(1,lastrow - firstrow + 2);
    NS::for_each(PARALLEL,jrange.begin(), jrange.end(), [&](auto j) -> void {
      auto k2=view::ints(rowstr[j],rowstr[j + 1]);
      double d=std::accumulate(k2.begin(),k2.end(),0.0,[=](double i,int k)->double{
        return i+a[k] * z[colidx[k]];
      });
      r[j] = d;
    });

/*--------------------------------------------------------------------
c  At this point, r contains A.z
c-------------------------------------------------------------------*/
    jrange=view::ints(1, lastcol - firstcol + 2);
    sum=std::accumulate(jrange.begin(),jrange.end(),0.0,[=](double i,int j)->double{
      double d = x[j] - r[j];
      return i+d*d;
    });
  (*rnorm) = sqrt(sum);
}

/*---------------------------------------------------------------------
c       generate the test problem for benchmark 6
c       makea generates a sparse matrix with a
c       prescribed sparsity distribution
c
c       parameter    type        usage
c
c       input
c
c       n            i           number of cols/rows of matrix
c       nz           i           nonzeros as declared array size
c       rcond        r*8         condition number
c       shift        r*8         main diagonal shift
c
c       output
c
c       a            r*8         array for nonzeros
c       colidx       i           col indices
c       rowstr       i           row pointers
c
c       workspace
c
c       iv, arow, acol i
c       v, aelt        r*8
c---------------------------------------------------------------------*/
void makea(int n, int nz, double a[], /* a[1:nz] */
                  int colidx[],              /* colidx[1:nz] */
                  int rowstr[],              /* rowstr[1:n+1] */
                  int nonzer, int firstrow, int lastrow, int firstcol,
                  int lastcol, double rcond, int arow[], /* arow[1:nz] */
                  int acol[],                            /* acol[1:nz] */
                  double aelt[],                         /* aelt[1:nz] */
                  double v[],                            /* v[1:n+1] */
                  int iv[],                              /* iv[1:2*n+1] */
                  double shift) {
  int i, nnza, iouter, ivelt, ivelt1, irow, nzv;

  /*--------------------------------------------------------------------
  c      nonzer is approximately  (int(sqrt(nnza /n)));
  c-------------------------------------------------------------------*/

  double size, ratio, scale;
  int jcol;

  size = 1.0;
  ratio = pow(rcond, (1.0 / (double)n));
  nnza = 0;

/*---------------------------------------------------------------------
c  Initialize colidx(n+1 .. 2n) to zero.
c  Used by sprnvc to mark nonzero positions
c---------------------------------------------------------------------*/
auto irange=view::ints(1,n+1);
  NS::for_each(PARALLEL,irange.begin(), irange.end(), [&](auto i) -> void {
    colidx[n + i] = 0;
  });
  for (iouter = 1; iouter <= n; iouter++) {
    nzv = nonzer;
    sprnvc(n, nzv, v, iv, &(colidx[0]), &(colidx[n]));
    vecset(n, v, iv, &nzv, iouter, 0.5);
    for (ivelt = 1; ivelt <= nzv; ivelt++) {
      jcol = iv[ivelt];
      if (jcol >= firstcol && jcol <= lastcol) {
        scale = size * v[ivelt];
        for (ivelt1 = 1; ivelt1 <= nzv; ivelt1++) {
          irow = iv[ivelt1];
          if (irow >= firstrow && irow <= lastrow) {
            nnza = nnza + 1;
            if (nnza > nz) {
              printf("Space for matrix elements exceeded in"
                     " makea\n");
              printf("nnza, nzmax = %d, %d\n", nnza, nz);
              printf("iouter = %d\n", iouter);
              exit(1);
            }
            acol[nnza] = jcol;
            arow[nnza] = irow;
            aelt[nnza] = v[ivelt1] * scale;
          }
        }
      }
    }
    size = size * ratio;
  }

  /*---------------------------------------------------------------------
  c       ... add the identity * rcond to the generated matrix to bound
  c           the smallest eigenvalue from below by rcond
  c---------------------------------------------------------------------*/
  for (i = firstrow; i <= lastrow; i++) {
    if (i >= firstcol && i <= lastcol) {
      iouter = n + i;
      nnza = nnza + 1;
      if (nnza > nz) {
        printf("Space for matrix elements exceeded in makea\n");
        printf("nnza, nzmax = %d, %d\n", nnza, nz);
        printf("iouter = %d\n", iouter);
        exit(1);
      }
      acol[nnza] = i;
      arow[nnza] = i;
      aelt[nnza] = rcond - shift;
    }
  }

  /*---------------------------------------------------------------------
  c       ... make the sparse matrix from list of elements with duplicates
  c           (v and iv are used as  workspace)
  c---------------------------------------------------------------------*/
  sparse(a, colidx, rowstr, n, arow, acol, aelt, firstrow, lastrow, v, &(iv[0]),
         &(iv[n]), nnza);
}

/*---------------------------------------------------
c       generate a sparse matrix from a list of
c       [col, row, element] tri
c---------------------------------------------------*/
void sparse(double a[],                            /* a[1:*] */
                   int colidx[],                          /* colidx[1:*] */
                   int rowstr[],                          /* rowstr[1:*] */
                   int n, int arow[],                     /* arow[1:*] */
                   int acol[],                            /* acol[1:*] */
                   double aelt[],                         /* aelt[1:*] */
                   int firstrow, int lastrow, double x[], /* x[1:n] */
                   boolean mark[],                        /* mark[1:n] */
                   int nzloc[],                           /* nzloc[1:n] */
                   int nnza)
/*---------------------------------------------------------------------
c       rows range from firstrow to lastrow
c       the rowstr pointers are defined for nrows = lastrow-firstrow+1 values
c---------------------------------------------------------------------*/
{
  int nrows;
  int i, j, jajp1, nza, k, nzrow;
  double xi;
  auto jrange = view::ints(1, n + 1);
  auto irange = view::ints(0, 1);

  /*--------------------------------------------------------------------
  c    how many rows of result
  c-------------------------------------------------------------------*/
  nrows = lastrow - firstrow + 1;

/*--------------------------------------------------------------------
c     ...count the number of triples in each row
c-------------------------------------------------------------------*/
jrange = view::ints(1, n+1);
NS::for_each(PARALLEL,jrange.begin(), jrange.end(), [&](auto j) -> void {
    rowstr[j] = 0;
    mark[j] = FALSE;
  });
  rowstr[n + 1] = 0;

  for (nza = 1; nza <= nnza; nza++) {
    j = (arow[nza] - firstrow + 1) + 1;
    rowstr[j] = rowstr[j] + 1;
  }

  rowstr[1] = 1;
  for (j = 2; j <= nrows + 1; j++) {
    rowstr[j] = rowstr[j] + rowstr[j - 1];
  }

  /*---------------------------------------------------------------------
  c     ... rowstr(j) now is the location of the first nonzero
  c           of row j of a
  c---------------------------------------------------------------------*/

  /*---------------------------------------------------------------------
  c     ... preload data pages
  c---------------------------------------------------------------------*/
  jrange = view::ints(0, nrows);
  NS::for_each(PARALLEL,jrange.begin(), jrange.end(), [&](auto j) -> void {
    auto k2range = view::ints(rowstr[j], rowstr[j + 1]);
    NS::for_each(PARALLELUNSEQ,k2range.begin(), k2range.end(),
                  [&](auto k) -> void { a[k] = 0.0; });
  });
  /*--------------------------------------------------------------------
  c     ... do a bucket sort of the triples on the row index
  c-------------------------------------------------------------------*/
  auto nzarange = view::ints(1, nnza + 1);
  NS::for_each(PARALLELUNSEQ,nzarange.begin(), nzarange.end(), [&](auto nza) -> void {
    int j = arow[nza] - firstrow + 1;
    int k = rowstr[j];
    a[k] = aelt[nza];
    colidx[k] = acol[nza];
    rowstr[j] = rowstr[j] + 1;
  });

  /*--------------------------------------------------------------------
  c       ... rowstr(j) now points to the first element of row j+1
  c-------------------------------------------------------------------*/
  for (j = nrows; j >= 1; j--) {
    rowstr[j + 1] = rowstr[j];
  }
  rowstr[1] = 1;

  /*--------------------------------------------------------------------
  c       ... generate the actual output rows by adding elements
  c-------------------------------------------------------------------*/
  nza = 0;
  irange = view::ints(1, n+1);
  NS::for_each(PARALLELUNSEQ,irange.begin(), irange.end(), [&](auto i) -> void {
    x[i] = 0.0;
    mark[i] = FALSE;
  });

  jajp1 = rowstr[1];
  for (j = 1; j <= nrows; j++) {
    nzrow = 0;

    /*--------------------------------------------------------------------
    c          ...loop over the jth row of a
    c-------------------------------------------------------------------*/
    for (k = jajp1; k < rowstr[j + 1]; k++) {
      i = colidx[k];
      x[i] = x[i] + a[k];
      if (mark[i] == FALSE && x[i] != 0.0) {
        mark[i] = TRUE;
        nzrow = nzrow + 1;
        nzloc[nzrow] = i;
      }
    }

    /*--------------------------------------------------------------------
    c          ... extract the nonzeros of this row
    c-------------------------------------------------------------------*/
    for (k = 1; k <= nzrow; k++) {
      i = nzloc[k];
      mark[i] = FALSE;
      xi = x[i];
      x[i] = 0.0;
      if (xi != 0.0) {
        nza = nza + 1;
        a[nza] = xi;
        colidx[nza] = i;
      }
    }
    jajp1 = rowstr[j + 1];
    rowstr[j + 1] = nza + rowstr[1];
  }
}

/*---------------------------------------------------------------------
c       generate a sparse n-vector (v, iv)
c       having nzv nonzeros
c
c       mark(i) is set to 1 if position i is nonzero.
c       mark is all zero on entry and is reset to all zero before exit
c       this corrects a performance bug found by John G. Lewis, caused by
c       reinitialization of mark on every one of the n calls to sprnvc
---------------------------------------------------------------------*/
void sprnvc(int n, int nz, double v[], /* v[1:*] */
                   int iv[],                  /* iv[1:*] */
                   int nzloc[],               /* nzloc[1:n] */
                   int mark[])                /* mark[1:n] */
{
  int nn1;
  int nzrow, nzv, ii, i;
  double vecelt, vecloc;

  nzv = 0;
  nzrow = 0;
  nn1 = 1;
  do {
    nn1 = 2 * nn1;
  } while (nn1 < n);

  /*--------------------------------------------------------------------
  c    nn1 is the smallest power of two not less than n
  c-------------------------------------------------------------------*/

  while (nzv < nz) {
    vecelt = randlc(&tran, amult);

    /*--------------------------------------------------------------------
    c   generate an integer between 1 and n in a portable manner
    c-------------------------------------------------------------------*/
    vecloc = randlc(&tran, amult);
    i = icnvrt(vecloc, nn1) + 1;
    if (i > n)
      continue;

    /*--------------------------------------------------------------------
    c  was this integer generated already?
    c-------------------------------------------------------------------*/
    if (mark[i] == 0) {
      mark[i] = 1;
      nzrow = nzrow + 1;
      nzloc[nzrow] = i;
      nzv = nzv + 1;
      v[nzv] = vecelt;
      iv[nzv] = i;
    }
  }

  for (ii = 1; ii <= nzrow; ii++) {
    i = nzloc[ii];
    mark[i] = 0;
  }
}

/*---------------------------------------------------------------------
 * scale a double precision number x in (0,1) by a power of 2 and chop it
 *---------------------------------------------------------------------*/
static int icnvrt(double x, int ipwr2) { return ((int)(ipwr2 * x)); }

/*--------------------------------------------------------------------
c       set ith element of sparse vector (v, iv) with
c       nzv nonzeros to val
c-------------------------------------------------------------------*/
void vecset(int n, double v[], /* v[1:*] */
                   int iv[],          /* iv[1:*] */
                   int *nzv, int i, double val) {
  int k;
  boolean set;

  set = FALSE;
  for (k = 1; k <= *nzv; k++) {
    if (iv[k] == i) {
      v[k] = val;
      set = TRUE;
    }
  }
  if (set == FALSE) {
    *nzv = *nzv + 1;
    v[*nzv] = val;
    iv[*nzv] = i;
  }
}