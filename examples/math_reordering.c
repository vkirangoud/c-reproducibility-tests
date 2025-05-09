#include <stdio.h>

int main() {
    double a = 1e16;
    double b = -1e16;
    double c = 1.0;
    double result = (a + b) + c;
    printf("%.17g\n", result);
    return 0;
}
// This program demonstrates the effect of floating-point arithmetic reordering.
// The result may vary depending on the order of operations due to the limited precision
// of floating-point numbers. In this case, the result may be 1.0 or 0.0 depending on the
// compiler and optimization settings. This is a classic example of how floating-point
// arithmetic can lead to non-intuitive results due to the way numbers are represented
// in memory and the order in which operations are performed.
// To compile and run this program, use the following commands:
// gcc -o math_reordering math_reordering.c
// ./math_reordering