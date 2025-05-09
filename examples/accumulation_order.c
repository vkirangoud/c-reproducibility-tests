#include <stdio.h>

int main() {
    float sum = 0.0f;
    for (int i = 0; i < 1000; i++)
        sum += 0.1f;
    printf("%.17g\n", sum);
    return 0;
}
