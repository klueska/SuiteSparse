// =============================================================================
// === qrdemo.cpp ==============================================================
// =============================================================================

// A simple C++ demo of SuiteSparseQR.  The comments give the MATLAB equivalent
// statements.  See also qrdemo.m

#include "SuiteSparseQR.hpp"
#include <complex>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

// SuiteSparseQR uses an integer defined in SuiteSparse_config.h called
// SuiteSparse_long.  It is a 32-bit integer on a 32-bit platform, and a 64-bit
// integer on a 64-bit platform.  For most platforms (except Windows),
// SuiteSparse_long is just "long".

#define Long SuiteSparse_long

// =============================================================================
// check_residual:  print the relative residual, norm (A*x-b)/norm(x)
// =============================================================================

void check_residual
(
    cholmod_sparse *A,
    cholmod_dense *X,
    cholmod_dense *B,
    cholmod_common *cc
)
{
    Long m = A->nrow ;
    Long n = A->ncol ;
    Long rnk ;
    double rnorm, anorm, xnorm ;
    double one [2] = {1,0}, minusone [2] = {-1,0} ;
    cholmod_dense *Residual ;

    // get the rank(A) estimate
    rnk = cc->SPQR_istat [4] ;

    // anorm = norm (A,1) ;
    anorm = cholmod_l_norm_sparse (A, 1, cc) ;

    // rnorm = norm (A*X-B)
    Residual = cholmod_l_copy_dense (B, cc) ;
    cholmod_l_sdmult (A, 0, one, minusone, X, Residual, cc) ;
    rnorm = cholmod_l_norm_dense (Residual, 2, cc) ;

    // xnorm = norm (X)
    xnorm = cholmod_l_norm_dense (X, 2, cc) ;

    if (m <= n && anorm > 0 && xnorm > 0)
    {
        // find the relative residual, except for least-squares systems
        rnorm /= (anorm * xnorm) ;
    }
    printf ("residual: %8.1e rank: %6ld\n", rnorm, rnk) ;
    cholmod_l_free_dense (&Residual, cc) ;
}

// =============================================================================

int main (int argc, char **argv)
{
    cholmod_common Common, *cc ;
    cholmod_sparse *A ;
    cholmod_dense *X, *B ;
    int mtype ;
    Long m, n ;
    char* grain, *nthreads ;

    // start CHOLMOD
    cc = &Common ;
    cholmod_l_start (cc) ;

    // Set the number of TBB threads, if not set, it gets set to -1, which will
    // trigger us to look for SPQR_GRAIN to decide what to do for parallelism.
    nthreads = getenv("TBB_NUM_THREADS");
    if (nthreads == NULL)
        cc->SPQR_nthreads = -1;
    else if (strcmp(nthreads, "BEST") == 0)
        cc->SPQR_nthreads = 0;
    else
        cc->SPQR_nthreads = atof(nthreads);

    // If SPQR_nthreads is explicitly set to 0 or set to BEST, then
    // automatically set the grain size to 2* the number of processors in the
    // system (as recommended in the SPQR documentation).
    if (cc->SPQR_nthreads == 0) {
        cc->SPQR_grain = sysconf(_SC_NPROCESSORS_ONLN) * 2;
    // Otherwise, if it is explicitly set to > 0, then automatically set the
    // grain size to 2* the number of threads specified (as recommended in the
    // SPQR documentation).
    } else if (cc->SPQR_nthreads > 0) {
        cc->SPQR_grain = cc->SPQR_nthreads * 2;
    // Otherwise, use SQPR_GRAIN to set the grain size to use for TBB
    // parallelism. The number of threads is determined automatically by TBB.
    // If SPQR_GRAIN is not set, default it 1.  If set to MAX, set it to twice
    // the number of processors on the system (as recommended in the SPQR
    // documentation).
    } else {
        grain = getenv("SPQR_GRAIN");
        if (grain == NULL)
            cc->SPQR_grain = 1;
        else if (strcmp(grain, "MAX") == 0)
            cc->SPQR_grain = sysconf(_SC_NPROCESSORS_ONLN) * 2;
        else
            cc->SPQR_grain = atof(grain);

        if (cc->SPQR_grain < 1) {
            printf ("SPQR grain size must be >= 1\n");
            exit (1);
        }
    }

    // A = mread (stdin) ; read in the sparse matrix A
    A = (cholmod_sparse *) cholmod_l_read_matrix (stdin, 1, &mtype, cc) ;
    if (mtype != CHOLMOD_SPARSE)
    {
        printf ("input matrix must be sparse\n") ;
        exit (1) ;
    }

    // [m n] = size (A) ;
    m = A->nrow ;
    n = A->ncol ;

    printf ("Matrix %6ld-by-%-6ld nnz: %6ld\n", m, n, cholmod_l_nnz (A, cc)) ;

    // B = ones (m,1), a dense right-hand-side of the same type as A
    B = cholmod_l_ones (m, 1, A->xtype, cc) ;

    // X = A\B ; with default ordering and default column 2-norm tolerance
    if (A->xtype == CHOLMOD_REAL)
    {
        // A, X, and B are all real
        X = SuiteSparseQR <double>
            (SPQR_ORDERING_DEFAULT, SPQR_DEFAULT_TOL, A, B, cc) ;
    }
    else
    {
        // A, X, and B are all complex
        X = SuiteSparseQR < std::complex<double> >
            (SPQR_ORDERING_DEFAULT, SPQR_DEFAULT_TOL, A, B, cc) ;
    }

    check_residual (A, X, B, cc) ;
    cholmod_l_free_dense (&X, cc) ;

    // -------------------------------------------------------------------------
    // factorizing once then solving twice with different right-hand-sides
    // -------------------------------------------------------------------------

    // Just the real case.  Complex case is essentially identical
    if (A->xtype == CHOLMOD_REAL)
    {
        SuiteSparseQR_factorization <double> *QR ;
        cholmod_dense *Y ;
        Long i ;
        double *Bx ;

        // factorize once
        QR = SuiteSparseQR_factorize <double>
            (SPQR_ORDERING_DEFAULT, SPQR_DEFAULT_TOL, A, cc) ;

        // solve Ax=b, using the same B as before

        // Y = Q'*B
        Y = SuiteSparseQR_qmult (SPQR_QTX, QR, B, cc) ;
        // X = R\(E*Y)
        X = SuiteSparseQR_solve (SPQR_RETX_EQUALS_B, QR, Y, cc) ;
        // check the results
        check_residual (A, X, B, cc) ;
        // free X and Y
        cholmod_l_free_dense (&Y, cc) ;
        cholmod_l_free_dense (&X, cc) ;

        // repeat with a different B
        Bx = (double *) (B->x) ;
        for (i = 0 ; i < m ; i++)
        {
            Bx [i] = i ;
        }

        // Y = Q'*B
        Y = SuiteSparseQR_qmult (SPQR_QTX, QR, B, cc) ;
        // X = R\(E*Y)
        X = SuiteSparseQR_solve (SPQR_RETX_EQUALS_B, QR, Y, cc) ;
        // check the results
        check_residual (A, X, B, cc) ;
        // free X and Y
        cholmod_l_free_dense (&Y, cc) ;
        cholmod_l_free_dense (&X, cc) ;

        // free QR
        SuiteSparseQR_free (&QR, cc) ;
    }

    // -------------------------------------------------------------------------
    // free everything that remains
    // -------------------------------------------------------------------------

    cholmod_l_free_sparse (&A, cc) ;
    cholmod_l_free_dense (&B, cc) ;
    cholmod_l_finish (cc) ;
    return (0) ;
}
