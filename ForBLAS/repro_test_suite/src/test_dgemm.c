#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>

int main(int argc, char *argv[]) {
    const int M = 50, N = 60, K = 40;
    const double alpha = 1.0, beta = 1.0;
    double *A = malloc(M*K*sizeof(double));
    double *B = malloc(K*N*sizeof(double));
    double *C = malloc(M*N*sizeof(double));
    if (!A||!B||!C) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    // Use a fixed seed for reproducibility
    srand(1234);
    for (int i = 0; i < M*K; i++) A[i] = (double)rand()/RAND_MAX;
    for (int i = 0; i < K*N; i++) B[i] = (double)rand()/RAND_MAX;
    for (int i = 0; i < M*N; i++) C[i] = (double)rand()/RAND_MAX;  // initial C

    // Perform matrix multiplication: C = alpha*A*B + beta*C
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, alpha, A, K, B, N, beta, C, N);

    // Write output C matrix to binary file
    const char *outfile = (argc > 1) ? argv[1] : "dgemm_output.bin";
    FILE *f = fopen(outfile, "wb");
    fwrite(C, sizeof(double), M*N, f);
    fclose(f);

    free(A); free(B); free(C);
    return 0;
}
