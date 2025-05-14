#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>

int main(int argc, char *argv[]) {
    const int N = 1000;
    const double alpha = 2.0;
    double *x = malloc(N*sizeof(double));
    double *y = malloc(N*sizeof(double));
    if (!x||!y) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    srand(1234);
    for (int i = 0; i < N; i++) {
        x[i] = (double)rand()/RAND_MAX;
        y[i] = (double)rand()/RAND_MAX;
    }

    // Perform y = alpha * x + y
    cblas_daxpy(N, alpha, x, 1, y, 1);

    // Write output vector y to binary file
    const char *outfile = (argc > 1) ? argv[1] : "daxpy_output.bin";
    FILE *f = fopen(outfile, "wb");
    fwrite(y, sizeof(double), N, f);
    fclose(f);

    free(x); free(y);
    return 0;
}
