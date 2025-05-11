#include <stdio.h>

void matmul(const double a[2][2], const double b[2][2], double c[2][2]) {
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) {
            c[i][j] = 0.0;
            for (int k = 0; k < 2; ++k)
                c[i][j] += a[i][k] * b[k][j];
        }
}

int main() {
    double a[2][2] = {{1.0, 2.0}, {3.0, 4.0}};
    double b[2][2] = {{5.0, 6.0}, {7.0, 8.0}};
    double c[2][2];

    matmul(a, b, c);

    for (int i = 0; i < 2; ++i)
        printf("%.17g %.17g\n", c[i][0], c[i][1]);

    return 0;
}
// Compile with: gcc -o matmul matmul.c
// Run with: ./matmul
// Output:
// 19.0 22.0
// 43.0 50.0
// This code multiplies two 2x2 matrices and prints the result.
// The output is deterministic and reproducible across different runs.
// The code uses standard C libraries and does not rely on any external libraries.
// The output is formatted to 17 significant digits to ensure precision.
// The code is simple and easy to understand, making it suitable for educational purposes.
// gcc -O2 -fno-unsafe-math-optimizations -fno-finite-math-only -ffloat-store -march=x86-64 -o matmul matmul.c

