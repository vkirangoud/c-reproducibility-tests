# C Reproducibility Tests

This repository contains a set of **C code examples** to demonstrate common **bit reproducibility challenges** and solutions, particularly when dealing with floating-point operations, compiler optimizations, parallelism, and randomness.

### 1. Build the programs:

```bash
make
```
## 2. Running the Tests
```bash
./run_tests.sh
```
The first time each program is run, a reference output will be saved. On subsequent runs, it will compare the new output with the saved reference to check for bitwise differences.

## 3. Example Tests
- Accumulation Order: Demonstrates floating-point precision issues due to the order of accumulation in loops.

- x87 Precision: Shows issues with 80-bit extended precision in x86 CPUs.

- Math Reordering: Demonstrates how compiler optimizations can reorder math expressions and affect results.

- OpenMP Sum: Shows nondeterminism in parallel floating-point summation.

- Random Seed: Demonstrates randomness without a fixed seed, leading to different outputs across runs.
## 4. How to Contribute
Feel free to open issues or submit pull requests if you encounter other reproducibility challenges or solutions.

---

### 📄 **5. C Source Files**

#### **`examples/accumulation_order.c`**

```c
#include <stdio.h>

int main() {
    float sum = 0.0f;
    for (int i = 0; i < 1000; i++)
        sum += 0.1f;
    printf("%.17g\n", sum);
    return 0;
}
```


