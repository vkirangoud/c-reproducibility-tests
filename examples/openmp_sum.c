#include <stdio.h>
#include <omp.h>

int main() {
    float sum = 0.0f;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < 1000000; i++)
        sum += 0.1f;
    printf("%.17g\n", sum);
    return 0;
}
// Compile with: gcc -fopenmp -o openmp_sum openmp_sum.c
// Run with: OMP_NUM_THREADS=4 ./openmp_sum
// Expected output: 100000.00000000000000
// Note: The output may vary slightly due to floating-point precision and the number of threads used.
// This code demonstrates the use of OpenMP for parallel reduction.
// It calculates the sum of 0.1 added 1,000,000 times in parallel using multiple threads.
// The reduction clause ensures that the sum variable is correctly updated across threads.
// The expected output is 100000.00000000000000, but it may vary slightly due to floating-point precision.
// The code is a simple example of using OpenMP to parallelize a loop that computes the sum of a series of floating-point numbers.
// The OpenMP parallel for directive is used to distribute the iterations of the loop across multiple threads.
// The reduction clause specifies that the sum variable should be treated as a reduction variable,
// meaning that each thread will have its own private copy of the variable, and at the end of the parallel region,
// the private copies will be combined to produce the final result.