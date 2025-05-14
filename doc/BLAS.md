# AOCL‑BLAS and Cross‑CPU Consistency 
AOCL-BLAS is bit-reproducible across different AMD hardware configurations implies checking if BLAS operations using AOCL-BLAS produce identical binary outputs on various AMD CPUs, under the same software and build settings.
AMD’s AOCL‑BLAS is a tuned BLIS-based library for Zen CPUs, dynamically selecting CPU‑specific kernels at run time.  By default it chooses the “best” kernel for each processor (e.g. a Zen4-optimized GEMM on Genoa vs a Zen3 one on Milan), so identical code can produce slightly different floating‑point results on different AMD chips. In fact, documentation (and MathWorks’ guidelines) note that *bitwise* reproducibility holds only if the hardware (CPU microarchitecture, instruction set, number of threads, etc.) is unchanged. Changing the CPU family or BLAS backend typically leads to small round‑off differences.  Similarly, multithreaded execution can reorder summations or use atomic reductions, which can break bit-for-bit identity across runs.  In short, AOCL‑BLAS **does not guarantee** identical  results on EPYC vs Ryzen under normal operation.

However, AOCL‑BLAS does provide knobs to enforce a uniform code path.  For example, AMD’s AOCL documentation states that setting the environment variable `BLIS_ARCH_TYPE` (to one of `{zen4, zen3, zen2, zen, generic}`) will “completely override” the automatic dispatch.  This forces the library to use a single architecture’s kernels (for instance, specifying `zen3` on both platforms).  In practice, you can pick the lowest common denominator (e.g. `zen2` or even `generic`) so that both CPUs run the *same* code path.  Likewise, `BLIS_MODEL_TYPE={Milan, Genoa, …}` can be used to pick a particular processor model (Milan/Milan‑X=Zen3, Genoa/Bergamo=Zen4).  Using these variables in concert can yield *nearly* identical results: AMD notes that setting a fixed model/arch “may provide consistent results across different models if consistency is a higher priority than best performance”.  (Note that forcing an incompatible code path will cause errors – e.g. Zen4 kernels on a Zen3 CPU can illegal-instruct.)

**Sources of Variation:** In summary, even with the same AOCL version and data, differences arise from (1) **architecture‑specific kernels** (vector width, FMA usage, instruction latency, etc.), (2) **threading and reduction order** (atomics or OpenMP sum order), and (3) **runtime dispatch logic**.  For example, AOCL‑BLAS may use AVX2 on one CPU and AVX512 on another, or execute inner loops in a different associative order.  Multi‑threaded summations are generally not guaranteed to be bit‑for‑bit reproducible.  Indeed, MathWorks explicitly warns that identical floating‑point outputs only occur if CPU type and BLAS library are unchanged.  If strict determinism is needed, one must manually control these factors.

**Reproducibility Controls:**  To maximize consistency, use AOCL/BLIS environment options and single‑threading:

* **Force one architecture path:** Set `BLIS_ARCH_TYPE` (e.g. “zen3”) so both machines run the same kernel code. A “generic” setting uses C reference code (portable but slow) for truly uniform behavior.
* **Fix the CPU model:** Set `BLIS_MODEL_TYPE` to the same model name (Milan, Genoa, etc.) on both systems to suppress model‑specific tweaks.
* **Limit threading:** Set `BLIS_NUM_THREADS=1` (or `OMP_NUM_THREADS=1`) or use the *single‑threaded* AOCL-BLAS library. This avoids nondeterminism from parallel reduction order. Pin threads or use static scheduling if multithreaded.
* **Compiler flags:** Ensure both systems use the same floating‑point compiler modes (disable mixed precision or flush subnormals in the same way) to avoid hidden differences.
* **Library versions:** Use the identical AOCL‑BLAS binary on both systems; dynamic dispatch will only differ if the CPU differs (else no difference in code).

In practice, many users report that with these settings AOCL-BLAS produces *nearly* identical results, but perfect bit-for-bit identity is hard to guarantee outside a single CPU model. AMD provides no one-switch “reproducible mode,” but the above tips can minimize variations. If ultimate reproducibility is required, using the generic (pure-C) code path and single thread on both machines will ensure identical bitwise output – at the cost of performance.

**Practical Guidance:** In summary, AOCL-BLAS will likely yield slightly different results on different AMD CPUs unless carefully configured. To test, you can run AOCL’s own DGEMM tests or your workload with `BLIS_ARCH_DEBUG=1` to see which sub-configuration is chosen.  If differences arise, try setting `BLIS_ARCH_TYPE` (or `BLIS_MODEL_TYPE`) identically on all machines. Also compare single-threaded vs multi-threaded runs. Understanding that FP arithmetic is deterministic only for a given code path, the safest approach for strict reproducibility is to **freeze the code path and threads** as above. In most cases small differences are acceptable, but for true bit-for-bit need, use a common code path (e.g. “generic”) and a single thread.

**Sources:** AMD’s AOCL documentation describes the dynamic dispatch and environment variables for BLIS (AOCL-BLAS).  MathWorks notes that floating-point results only match exactly if CPU/threads/BLAS are unchanged. These sources (and user reports) indicate that AOCL-BLAS does *not* automatically guarantee bitwise reproducibility across different AMD chips, but it can be tuned via BLIS environment variables to achieve it.

Great. I’ll prepare reproducibility test scripts for the AOCL-BLAS functions `dgemm`, `daxpy`, and `dot`, focused on ensuring bit-for-bit identical results across AMD systems.

The test suite will include examples, comparison logic, and configuration settings to maximize reproducibility. I’ll share the full setup shortly.


# Reproducibility Test Suite for AOCL-BLAS (BLIS)

To ensure **bit-for-bit** reproducibility of AOCL-BLAS (BLIS) routines across AMD CPUs (e.g. Ryzen, EPYC), we create C test programs for `dgemm`, `daxpy`, and `dot` that use fixed random seeds, output raw binary results, and compare them against reference outputs.  The suite supports both single-threaded and multi-threaded modes, and honors the `BLIS_ARCH_TYPE`/`BLIS_MODEL_TYPE` environment variables for architecture dispatch.  Below is an example directory layout and sample code with a `Makefile` and runner script to automate the tests.

```
repro_test_suite/
│
├── Makefile
├── run_tests.sh
├── ref/                      # (Contains reference output binaries)
│   ├── dgemm_single.ref
│   ├── dgemm_multi.ref
│   ├── daxpy_single.ref
│   ├── daxpy_multi.ref
│   ├── dot_single.ref
│   └── dot_multi.ref
└── src/
    ├── test_dgemm.c
    ├── test_daxpy.c
    └── test_dot.c
```

* **`src/`**: Contains the C test programs.
* **`Makefile`**: Builds each test program and links against AOCL-BLAS (BLIS) libraries (single-thread and multi-thread versions).
* **`run_tests.sh`**: Shell script to execute tests, generate reference outputs if absent, and compare outputs bitwise using `cmp` or `sha256sum`.
* **`ref/`**: Stores “golden” reference output files (generated on one CPU) for comparison on subsequent runs.

## Test Program Code

Each test program initializes input data with a fixed seed (`srand(1234)`), calls the appropriate BLAS routine, and writes the raw double-precision result(s) to a binary file.  This ensures that running the test on different CPUs (under the same environment) should produce *identical bits* if the library is reproducible. Below are example implementations:

### `test_dgemm.c`

```c
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
```

This program creates random matrices **A**, **B**, and initial **C**, then calls `cblas_dgemm` (row-major) and writes the result matrix **C** to a binary file.  All values depend on the fixed `srand` seed, so they should be identical bitwise on each run.

### `test_daxpy.c`

```c
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
```

This program performs the BLAS `daxpy` operation with a fixed seed. It writes the resulting vector **y** to a binary file for later comparison.

### `test_dot.c`

```c
#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>

int main(int argc, char *argv[]) {
    const int N = 1000;
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

    // Compute dot product
    double result = cblas_ddot(N, x, 1, y, 1);

    // Write the double result to binary file
    const char *outfile = (argc > 1) ? argv[1] : "dot_output.bin";
    FILE *f = fopen(outfile, "wb");
    fwrite(&result, sizeof(double), 1, f);
    fclose(f);

    free(x); free(y);
    return 0;
}
```

The `dot` test calculates the dot product and writes the 8-byte `double` result to a file.  Again, the fixed seed ensures the same value is produced on each run.

## Makefile

The following `Makefile` compiles the tests and links them against AOCL-BLAS (BLIS).  We show both a single-threaded (linking `-lblis`) and a multi-threaded (`-lblis-mt` with `-fopenmp`) target. Adjust `AOCL_INC` and `AOCL_LIB` to point to your AOCL-BLAS include and library paths.  The example assumes dynamic linking and GCC.

```makefile
AOCL_INC = /opt/amd/aocl/include   # adjust to your AOCL install path
AOCL_LIB = /opt/amd/aocl/lib       # adjust to your AOCL install path

CFLAGS = -O2 -I$(AOCL_INC)
LDFLAGS_SINGLE = -L$(AOCL_LIB) -lblis -lm -lpthread
LDFLAGS_MULTI  = -L$(AOCL_LIB) -lblis-mt -lm -fopenmp

# Default: build both single-threaded and multi-threaded binaries
all: test_dgemm_st test_daxpy_st test_dot_st test_dgemm_mt test_daxpy_mt test_dot_mt

test_dgemm_st: src/test_dgemm.c
	$(CC) $(CFLAGS) src/test_dgemm.c -o test_dgemm_st $(LDFLAGS_SINGLE)

test_daxpy_st: src/test_daxpy.c
	$(CC) $(CFLAGS) src/test_daxpy.c -o test_daxpy_st $(LDFLAGS_SINGLE)

test_dot_st: src/test_dot.c
	$(CC) $(CFLAGS) src/test_dot.c -o test_dot_st $(LDFLAGS_SINGLE)

test_dgemm_mt: src/test_dgemm.c
	$(CC) $(CFLAGS) src/test_dgemm.c -o test_dgemm_mt $(LDFLAGS_MULTI)

test_daxpy_mt: src/test_daxpy.c
	$(CC) $(CFLAGS) src/test_daxpy.c -o test_daxpy_mt $(LDFLAGS_MULTI)

test_dot_mt: src/test_dot.c
	$(CC) $(CFLAGS) src/test_dot.c -o test_dot_mt $(LDFLAGS_MULTI)

clean:
	rm -f test_*.o test_*_st test_*_mt *.bin
```

> **Linking note:**  AOCL-BLAS provides separate libraries for single-threaded (`-lblis`) and multi-threaded (`-lblis-mt`) use. In dynamic builds, link with `-lblis` (plus `-lpthread`) for single-threaded binaries, or `-lblis-mt` (plus `-fopenmp`) for multi-threaded binaries.

## Test Runner Script

The `run_tests.sh` script executes each test binary, manages single- vs multi-thread modes, and compares outputs against stored references. It uses `cmp` (or `sha256sum`) for bitwise comparison and logs clear pass/fail messages.

```bash
#!/bin/bash
set -e

# Directories
mkdir -p ref
OUTDIR="./outputs"
mkdir -p $OUTDIR

# Set environment (allow BLIS_ARCH_TYPE/BLIS_MODEL_TYPE override externally)
if [ -z "$BLIS_ARCH_TYPE" ]; then
    echo "Using default BLIS_ARCH_TYPE (no override)."
else
    echo "Using BLIS_ARCH_TYPE=$BLIS_ARCH_TYPE"
fi
if [ -z "$BLIS_MODEL_TYPE" ]; then
    echo "Using default BLIS_MODEL_TYPE (no override)."
else
    echo "Using BLIS_MODEL_TYPE=$BLIS_MODEL_TYPE"
fi

# Helper function to run a test and compare
run_test() {
    mode=$1     # "single" or "multi"
    prog=$2     # program name (without _st or _mt suffix)
    bin="$3"    # binary to run
    suffix=$4   # suffix for reference file ("single" or "multi")

    echo "Running $prog ($mode-thread)..."
    if [ "$mode" = "multi" ]; then
        export OMP_NUM_THREADS=4   # set multi-thread count
    else
        export OMP_NUM_THREADS=1
    fi

    outfile="$OUTDIR/${prog}.out.bin"
    ./"$bin" "$outfile"

    ref="./ref/${prog}_${suffix}.ref"
    if [ ! -f "$ref" ]; then
        # First-time reference creation
        echo "Creating reference for $prog (${mode}-thread)."
        cp "$outfile" "$ref"
    else
        # Compare output to reference
        if cmp -s "$outfile" "$ref"; then
            echo "$prog (${mode}-thread): PASS"
        else
            echo "$prog (${mode}-thread): FAIL"
            cmp "$outfile" "$ref" || true
        fi
    fi
    echo
}

# Build binaries if not built
if [ ! -x ./test_dgemm_st ]; then
    echo "Building test binaries..."
    make
fi

# Single-threaded tests
run_test "single" "dgemm" "test_dgemm_st" "single"
run_test "single" "daxpy" "test_daxpy_st" "single"
run_test "single" "dot"   "test_dot_st"   "single"

# Multi-threaded tests
run_test "multi" "dgemm" "test_dgemm_mt" "multi"
run_test "multi" "daxpy" "test_daxpy_mt" "multi"
run_test "multi" "dot"   "test_dot_mt"   "multi"

echo "All tests completed."
```

**Usage:** Make sure `test_*` binaries are built (e.g. by running `make`). Then execute:

```bash
./run_tests.sh
```

The script prints out which tests pass or fail. On the **first run**, it will generate reference files in `ref/`. On subsequent runs (especially on a different CPU), it will compare the outputs bit-for-bit with those references. A mismatch indicates a reproducibility issue.

## Environment Overrides

To test different code paths, you can set `BLIS_ARCH_TYPE` or `BLIS_MODEL_TYPE` before running. For example:

```bash
export BLIS_ARCH_TYPE=zen3
export BLIS_MODEL_TYPE=Milan
./run_tests.sh
```

This forces BLIS to use the specified architecture or CPU model code path.  The test suite itself does not parse these variables; it simply honors whatever BLIS path is activated by the environment. By comparing reference outputs across runs with different `BLIS_ARCH_TYPE`/`MODEL_TYPE` or on different AMD CPUs (e.g. Ryzen vs EPYC), you can verify bitwise consistency of the AOCL-BLAS routines.

### References

* AOCL-BLAS User Guide, AMD: instructions for linking with single-/multi-threaded BLIS libraries and using `BLIS_ARCH_TYPE`/`BLIS_MODEL_TYPE` for architecture dispatch.
* AOCL-BLAS example C code (demonstrates use of CBLAS API).
