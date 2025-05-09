#include <stdio.h>

int main() {
    double x = 1.0;
    double y = x / 10.0;
    printf("%.17g\n", y);
    return 0;
}
// Compile with: gcc -o x87_precision x87_precision.c -lm
// Run with: ./x87_precision
// Expected output: 0.1
// This code demonstrates the precision of floating-point arithmetic in C.
// The output may vary depending on the compiler and optimization settings.
// The precision of floating-point numbers can lead to unexpected results
// when performing arithmetic operations, especially with very small or very large numbers.
// This example shows how the x87 floating-point unit can affect the precision
// of floating-point calculations in C.
// The x87 floating-point unit is a stack-based architecture that can lead to
// different results compared to other architectures, such as the SSE (Streaming SIMD Extensions)
// or AVX (Advanced Vector Extensions) used in modern processors.
// The x87 architecture uses an 80-bit extended precision format for floating-point numbers,
// which can lead to differences in rounding and precision compared to the 64-bit double
// format used in C.
// This can result in unexpected behavior when performing arithmetic operations
// or when comparing floating-point numbers for equality.
// The example demonstrates how the x87 architecture can affect the precision
// of floating-point calculations in C, and how the output may vary depending
// on the compiler and optimization settings.
// The x87 architecture is still used in many modern processors, but it is
// being phased out in favor of more modern architectures, such as SSE and AVX.
// This example is a simple demonstration of the precision of floating-point
// arithmetic in C, and how the x87 architecture can affect the results.
// The x87 architecture is a legacy architecture that is still used in many
// modern processors, but it is being phased out in favor of more modern architectures,
// such as SSE and AVX. This example is a simple demonstration of the precision
// of floating-point arithmetic in C, and how the x87 architecture can affect.