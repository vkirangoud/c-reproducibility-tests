#include <stdio.h>
#include <stdlib.h>

int main() {
    for (int i = 0; i < 5; i++)
        printf("%d\n", rand());
    return 0;
}
// Compile with: gcc -o random_seed random_seed.c
// Run with: ./random_seed
// To reproduce the output, set the random seed before running the program.
// Example:
// $ export RANDFILE=/dev/urandom
// $ ./random_seed
// This will produce the same output every time you run it.
// To change the random seed, you can use the `srand` function.
// Example:
// #include <stdio.h>
// #include <stdlib.h>
// #include <time.h>
// int main() {
//    srand(time(NULL)); // Seed the random number generator with the current time
//    for (int i = 0; i < 5; i++)
//        printf("%d\n", rand());
//    return 0;
//}