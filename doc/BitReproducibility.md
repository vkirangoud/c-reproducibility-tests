# **Bit Reproducibility: Concepts, Challenges, and Practices**

**Author**: Kiran Varaganti  
**Date**: May 13, 2025  

---
**Bit reproducibility** (also called **bit-for-bit reproducibility** or **bitwise reproducibility**) means that a computational process, when run multiple times under the same conditions, produces *exactly* the same binary output every time — down to the last bit.

This is important in scientific computing, data analysis, and software development, especially for:

* **Validating results**: Ensuring that an experiment or model gives the exact same result when rerun.
* **Debugging**: Making sure that changes in behavior are due to code changes, not randomness.
* **Long-term reproducibility**: Letting others reproduce results years later.

To achieve bit reproducibility, you need:

* The **same software versions** and **libraries**.
* The **same hardware architecture** (or at least consistent floating-point behavior).
* **Deterministic code** (no randomness, or randomness controlled by a fixed seed).
* **Controlled environments** (like containers, or reproducible builds).

Here’s a practical example where **bit reproducibility** is critical:

### **Scientific Research – Climate Modeling**

Climate models simulate Earth’s climate system using vast amounts of math and physics. Researchers often run the same model across different supercomputers or rerun it later to verify findings or compare scenarios.

#### Scenario:

A research team develops a model to predict global temperature rise by 2100. They publish a paper, including their simulation results.

#### Why Bit Reproducibility Matters:

* **Verification**: Other researchers want to confirm the findings by rerunning the model. Bit reproducibility ensures they get *exactly* the same outputs.
* **Regulatory Impact**: Climate models influence policy decisions. If results change slightly due to non-reproducibility (e.g., from random rounding errors), it could undermine trust in the findings.
* **Debugging and Collaboration**: Team members working on different machines need to know any change in results is due to code edits, not system differences.

#### Without Bit Reproducibility:

* Slight floating-point differences across CPUs (e.g., Intel vs. AMD) or use of non-deterministic parallel threads can cause divergence over time in chaotic systems like climate models.
* These small variations can lead to completely different long-term predictions — undermining scientific rigor.

---

A well-known initiative that values bit reproducibility is **Reproducible Builds** ([https://reproducible-builds.org/](https://reproducible-builds.org/)), which aims to make software builds identical so they can be verified for integrity and security.

---
### In Context of System:

If a system is bit-reproducible, then:

* Running the same command (e.g. building software, training a model, or generating a file) multiple times, on the same platform and configuration, always produces **identical binary output**—not just functionally equivalent output, but **exact** byte-for-byte copies.

### Why It Matters:

* **Scientific reproducibility**: Ensures that computational research can be independently verified.
* **Security**: Guarantees that binaries correspond exactly to reviewed source code (e.g. in reproducible builds).
* **Debugging & consistency**: Makes it easier to identify subtle bugs and changes.

### Challenges to Achieving It:

* Timestamps embedded in files
* Non-deterministic algorithms (e.g., randomized training)
* Differences in compiler versions or environments
* Floating-point rounding differences on different hardware

### Example:

If you compile a program today and again next week, and the resulting binaries are **identical down to the last bit**, that is bit reproducibility.

Here are some tools and practices commonly used to achieve **bit reproducibility** in software builds and computational workflows:

---

### 🛠️ **Tools for Bit Reproducibility**

#### 1. **Reproducible Builds Toolchain**

* **[diffoscope](https://diffoscope.org/):** Compares files or directories to see why builds differ.
* **[reprotest](https://salsa.debian.org/reproducible-builds/reprotest):** Tests whether a build is reproducible under varying conditions.
* **[SOURCE\_DATE\_EPOCH](https://reproducible-builds.org/specs/source-date-epoch/):** Environment variable standard to replace timestamps in builds with a fixed date.

#### 2. **Build Systems**

* **[Nix / NixOS](https://nixos.org/):** Pure functional package manager; builds are highly deterministic and reproducible.
* **[Guix](https://guix.gnu.org/):** Functional package manager similar to Nix with emphasis on reproducibility.
* **[Bazel](https://bazel.build/):** Build system that emphasizes hermetic, sandboxed builds.

#### 3. **Containers & Sandboxes**

* **Docker (with caution):** Can help isolate environments, but needs care for true reproducibility.
* **Singularity / Apptainer:** Focused on scientific reproducibility with better integration for HPC environments.
* **ReproZip:** Captures the execution environment of experiments for reproducibility.

---

### ✅ **Best Practices for Bit Reproducibility**

1. **Control Timestamps**

   * Strip or standardize timestamps (e.g. via `SOURCE_DATE_EPOCH`).
   * Avoid embedding build time unless absolutely necessary.

2. **Fix Random Seeds**

   * Set random seeds explicitly in any stochastic process (e.g. `np.random.seed()` in Python).

3. **Lock Dependencies**

   * Pin exact versions of all packages (e.g., `requirements.txt`, `package-lock.json`, or Nix expressions).

4. **Use Deterministic Algorithms**

   * Choose algorithms that behave identically across runs (e.g. deterministic sorting, stable training modes).

5. **Normalize Locale and Timezone**

   * Set fixed environment variables like `LC_ALL=C` and `TZ=UTC`.

6. **Match System Environment**

   * Use the same OS, compiler, libraries, and system architecture.

---
### Bit Reproducibility Challenges between **AMD** and **Intel** CPUs
**Bit reproducibility across AMD vs Intel CPUs** is *very difficult* to achieve—sometimes even impossible—due to subtle differences in how these CPUs handle certain operations, especially:

---

### 🔍 **Why Bit Reproducibility May Fail Across AMD and Intel**

### 1. **Floating-Point Unit (FPU) Implementations**

* Intel and AMD may use different internal hardware designs to perform floating-point arithmetic.
* Even though both follow the **IEEE 754** standard (for floating-point precision), the way rounding, extended precision, or intermediate calculations are handled can differ slightly.

### 2. **Extended Precision (80-bit vs 64-bit)**

* Some Intel CPUs (especially older ones) perform intermediate calculations using **80-bit x87 floating-point registers**, even if the final result is 64-bit double.
* AMD CPUs (especially newer ones) might stick more closely to **64-bit SSE/AVX** operations, avoiding extended precision.
* These internal precision differences can lead to small rounding deviations in results.

### 3. **Instruction Scheduling and Optimizations**

* Different CPUs optimize instruction execution differently (e.g., pipelining, register allocation).
* When doing floating-point math at very high precision or large volumes, the order and timing of calculations can cause **tiny bit-level differences** due to rounding.

### 4. **Math Libraries and Compilers**

* If you're using standard math libraries (like `libm`, `math.h`, or compiler-specific intrinsics), these may behave slightly differently or be optimized differently for Intel vs. AMD.
* Compilers (like GCC, Clang, or Intel’s ICC) may emit different instructions based on detected architecture.

#### 5. **Floating-Point Arithmetic Differences**

* Even though both follow the IEEE 754 standard, implementation-level differences (e.g., in rounding, precision handling, fused multiply-add behavior) can cause **tiny numerical differences**, which break bit-for-bit reproducibility.
* These differences often show up in:

  * Scientific simulations
  * Machine learning training
  * Any code with accumulated floating-point operations

#### 6. **Instruction Set Variations**

* CPUs may use different micro-optimizations, SIMD instructions (e.g., AVX2 vs AVX-512), or fused instructions.
* These can change execution paths or timing in ways that affect numerical output.

#### 7. **Thread Scheduling and Parallelism**

* Even if using the same number of threads, OS-level thread scheduling and CPU behavior can lead to different execution orders, especially in parallel code—causing *non-determinism* unless explicitly controlled.

#### 8. **Compiler Behavior**

* Some compilers auto-detect CPU capabilities and emit optimized code differently depending on the hardware.
* Example: GCC and Clang may generate different machine code for the same source on AMD vs Intel.

---


### Example of a Difference:

Suppose you're computing:

```c
double a = 0.1;
double b = 0.2;
double c = a + b;
```

Even though mathematically `a + b = 0.3`, floating-point rounding might make `c == 0.30000000000000004` on one CPU and `c == 0.29999999999999999` on another — if the operations go through slightly different instruction paths or internal precision levels.

---

### Does It Matter?

* For most applications: **No**, these differences are negligible.
* For scientific simulations, cryptography, or bit-reproducible workflows: **Yes**, these can cause divergences.

Here’s a simple C code example that **can produce different floating-point results** on different CPUs (Intel vs AMD), or even on the *same* CPU depending on compiler settings or math optimizations:

---

### **Code: `float_diff.c`**

```c
#include <stdio.h>
#include <math.h>

int main() {
    volatile double a = 1.0e16;
    volatile double b = -1.0e16;
    volatile double c = 1.0;

    double result1 = (a + b) + c;
    double result2 = a + (b + c);

    printf("(a + b) + c = %.17g\n", result1);
    printf("a + (b + c) = %.17g\n", result2);

    return 0;
}
```

---

### **Expected Output:**

```text
(a + b) + c = 1
a + (b + c) = 0
```

---

### **Why This Happens:**

This is due to **floating-point associativity** not holding true:

* `(a + b) + c` becomes `(1e16 - 1e16) + 1 = 0 + 1 = 1`
* `a + (b + c)` becomes `1e16 + (-1e16 + 1) = 1e16 + 1 = 1e16` (but `1` is too small to affect `1e16`, so it gets lost)

---

### Now Imagine Running This On:

* **Intel**: Might use extended 80-bit precision registers by default, preserving more precision in intermediate results.
* **AMD**: Might use 64-bit SSE registers consistently, resulting in different rounding at each step.

These differences might not always show up for small-scale arithmetic, but in **large simulations or iterative algorithms**, such rounding variations can accumulate and diverge.

---

### To Test:

You can compile it with different flags to influence how rounding happens:

```bash
gcc -O2 float_diff.c -o float_diff
gcc -mfpmath=387 float_diff.c -o float_diff_x87     # uses x87 FPU (80-bit)
gcc -mfpmath=sse float_diff.c -o float_diff_sse     # forces SSE math (64-bit)
```

Then compare the outputs — they may differ subtly or significantly depending on hardware and compiler.

Making floating-point code more **consistent and reproducible across platforms** requires carefully controlling precision, math behavior, and compiler settings.

Here are **practical tips** to help:

---
### ✅ **How to Improve Reproducibility Across CPU Architectures**

| Strategy                                          | Description                                                                                                                                                                                                                                                                                                         |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 🧮 Use fixed-point instead of floating-point      | Eliminates rounding issues at the cost of flexibility.                                                                                                                                                                                                                                                              |
| ⚙️ Disable CPU-specific optimizations             | Use compiler flags like `-march=x86-64` instead of `-march=native`.                                                                                                                                                                                                                                                 |
| 🧵 Force single-threaded execution                | Avoids nondeterminism due to parallelism.                                                                                                                                                                                                                                                                           |
| 🔢 Use software libraries with strict determinism | e.g. [`numpy` with `np.seterr()` and fixed seeds](https://numpy.org/doc/stable/reference/random/generated/numpy.random.seed.html), or deterministic ML libraries like [PyTorch with `torch.use_deterministic_algorithms(True)`](https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html). |
| 🧪 Compare numerical output within tolerance      | Use `np.allclose()` instead of strict equality if only functional reproducibility is acceptable.                                                                                                                                                                                                                    |

---

### 🔬 Bottom Line

* **Bit-for-bit reproducibility between AMD and Intel CPUs is *not guaranteed***, especially for floating-point-heavy or multithreaded applications.
* However, it *is possible* to get very close or achieve **functionally equivalent** results with strict controls on randomness, compiler settings, and execution environment.

Here’s a **Bit Reproducibility Checklist** to help you test and improve reproducibility across AMD and Intel CPUs:

### ✅ **Bit Reproducibility Checklist (AMD vs Intel)**

#### 📦 **Environment Setup**

* [ ] **Same OS distribution & version** (e.g. Ubuntu 22.04 on both machines)
* [ ] **Same kernel version** (use `uname -r`)
* [ ] **Same compiler version** (e.g. GCC 12.3, Clang 16)
* [ ] **Same runtime libraries** (glibc, libm, etc.)
* [ ] **Disable CPU-specific build optimizations**

  * Use `-march=x86-64 -mtune=generic` instead of `-march=native`
  * In Python: avoid libraries that JIT based on CPU (e.g. numba with default settings)

#### 🧮 **Floating-Point Control**

* [ ] Use `-ffloat-store` or `-fno-fast-math` in GCC/Clang to minimize floating-point inconsistency
* [ ] Explicitly set math modes if available (e.g. fused multiply-add off)
* [ ] Use **software emulation** for floating-point if absolute reproducibility is critical

#### 🔢 **Randomness and Seeding**

* [ ] Set **fixed random seeds** in all libraries used

  * Python: `random.seed()`, NumPy: `np.random.seed()`, PyTorch: `torch.manual_seed()`
* [ ] Use `torch.use_deterministic_algorithms(True)` or similar for ML

#### 🧵 **Parallelism**

* [ ] Run code **single-threaded** (e.g., set `OMP_NUM_THREADS=1`)
* [ ] Or use libraries with deterministic parallel algorithms (e.g., joblib with `loky`)
* [ ] Use CPU affinity or taskset to pin process to specific cores

#### ⏱️ **Timestamps and File Metadata**

* [ ] Eliminate or standardize build timestamps using `SOURCE_DATE_EPOCH`
* [ ] Avoid filesystem-dependent metadata (like inode-based sorting or file creation time)

#### 🧪 **Output Comparison**

* [ ] Compare outputs with `sha256sum` or `cmp` for bit-level checks
* [ ] For numerical data, use `np.allclose()` with tight tolerances if perfect bit match isn't feasible

#### 📚 **Build Tools & Containers**

* [ ] Use reproducible build systems (e.g., Nix, Guix, Bazel)
* [ ] Or containerize environment (e.g., Docker, Apptainer) with fixed base images

---

### 🧰 Example Command Templates

**GCC/Clang Compilation Example:**

```bash
gcc -O2 -march=x86-64 -fno-unsafe-math-optimizations -ffloat-store -o myprog myprog.c
```
making floating-point code more **consistent and reproducible across platforms** requires carefully controlling precision, math behavior, and compiler settings.

Some more **practical tips** to help:

### **1. Use Fixed Precision (Avoid Extended Precision)**

* Avoid using **x87 FPU** which uses 80-bit intermediates by default.
* **Force SSE/AVX math**, which uses 64-bit IEEE-compliant operations:

```bash
gcc -mfpmath=sse -msse2 yourcode.c -o yourprog
```

This helps match behavior between Intel and AMD CPUs.

---

### **2. Avoid Unsafe Compiler Optimizations**

* Turn off aggressive floating-point optimizations that can re-order or approximate math:

```bash
gcc -O2 -fno-fast-math -ffloat-store yourcode.c
```

* `-ffloat-store`: Forces intermediate values to be stored in memory, limiting precision creep.
* `-fno-fast-math`: Disables assumptions like associativity or ignoring NaNs/infs.

---

### **3. Use Deterministic Math Libraries**

* Use math libraries designed for reproducibility, like:

  * [**crlibm** (Correctly Rounded Library for Mathematical Functions)](http://lipforge.ens-lyon.fr/www/crlibm/)
  * [**OpenLibm**](https://github.com/JuliaMath/openlibm)
* Avoid system-provided `libm` if exact bitwise matching is essential — some system math functions (e.g., `sin`, `log`) may differ across platforms.

---

### **4. Set the Floating-Point Environment**

Use standard functions to set consistent rounding modes:

```c
#include <fenv.h>
fesetround(FE_TONEAREST);   // or FE_DOWNWARD, FE_UPWARD, etc.
```

Compile with `-std=c99 -lm` and run with `#pragma STDC FENV_ACCESS ON` enabled if needed.

---

### **5. Fix Random Seeds and Threading**

If randomness is involved (e.g. in simulations or ML):

* Use a **fixed seed**: `srand(12345);`
* Avoid or control **parallel execution**, which may introduce nondeterministic ordering.

---

### **6. Use Software Emulation (if necessary)**

For full control and bit-for-bit reproducibility (especially for science):

* Use **MPFR** or **ARPREC** libraries for arbitrary-precision arithmetic (slower but exact).
* Use deterministic simulation frameworks (e.g., [REBOUND](https://rebound.readthedocs.io/) in astrophysics).

---



**Python Example:**

```python
import os, random, numpy as np, torch

os.environ["PYTHONHASHSEED"] = "0"
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)
```

**Run Single-Threaded:**

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
taskset -c 0 ./myprog  # pin to core 0
```

---
Great! Here's a **ready-to-use reproducibility setup** with:

1. A **Dockerfile** to create a fixed environment.
2. A **Python script** to demonstrate reproducible computation (including random number generation and matrix multiplication).

---

### 🐳 **Dockerfile: Reproducible Python Environment**

```Dockerfile
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONHASHSEED=0
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV SOURCE_DATE_EPOCH=1650000000
ENV TZ=UTC

# Install reproducibility-related tools and scientific packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    && pip install --no-cache-dir \
    numpy==1.24.0 \
    torch==2.0.1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY reproducible_script.py .

CMD ["python", "reproducible_script.py"]
```

---

### 🧪 **Python Script: `reproducible_script.py`**

```python
import os
import random
import numpy as np
import torch

# Set seeds and env vars for reproducibility
os.environ["PYTHONHASHSEED"] = "0"
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)

# Simple deterministic computation
print("Random integers:", random.randint(0, 100), random.randint(0, 100))
print("Numpy random vector:", np.random.rand(3))

# Torch matrix operation
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
c = torch.mm(a, b)
print("Torch matrix product:\n", c)

# Save output to verify bit-reproducibility
with open("output.txt", "w") as f:
    f.write(str(c.numpy()))
```

---

### 🚀 **Usage Instructions**

```bash
# Save Dockerfile and reproducible_script.py in the same folder
docker build -t reproducible-demo .
docker run --rm -v "$(pwd):/app" reproducible-demo
sha256sum output.txt
```

Run the same on **both AMD and Intel systems**—if everything works, the `output.txt` hash should be **identical**.

---
Here’s a **version with functional reproducibility** instead of requiring exact bit-for-bit matches. It compares floating-point outputs across runs and accepts small differences using `numpy.allclose()`.

---

### 🧪 **Updated Python Script: `reproducible_script_fuzzy.py`**

```python
import os
import random
import numpy as np
import torch

# Reproducibility setup
os.environ["PYTHONHASHSEED"] = "0"
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)

# Deterministic computation
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
result = torch.mm(a, b).numpy()

# Save result
np.savetxt("output.txt", result, fmt="%.10f")

# Optional: Load and compare to reference
reference_path = "reference_output.txt"
if os.path.exists(reference_path):
    reference = np.loadtxt(reference_path)
    identical = np.allclose(result, reference, rtol=1e-9, atol=1e-9)
    print("Result matches reference (within tolerance):", identical)
else:
    print("No reference file found; saving current output as reference.")
    np.savetxt(reference_path, result, fmt="%.10f")
```

---

### ✅ **Key Differences**

* Uses `np.allclose()` instead of `sha256sum`
* Allows small numerical differences (e.g., between AMD and Intel floating-point)
* Saves a reference file (`reference_output.txt`) to compare future runs

---

### 🔁 **How to Use**

```bash
docker build -t reproducible-demo-fuzzy .
docker run --rm -v "$(pwd):/app" reproducible-demo-fuzzy
```

* On the **first run**, it creates a reference file.
* On **subsequent runs**, it will compare the output to that file and report if it's close enough.

---
Bit reproducibility in **C** is especially challenging due to how close the language operates to the hardware, which exposes all the subtle, platform-dependent behaviors. Here's a breakdown of the **key challenges** and how to mitigate them:

---

### ⚠️ **Key Bit Reproducibility Challenges in C**

#### 1. 🧮 **Floating-Point Arithmetic**

* C uses platform-dependent floating-point behavior (IEEE 754, but with variations).
* Compilers may reorder operations or use extended precision (e.g., x87 80-bit registers on x86).
* Example problem:

  ```c
  double x = 0.1 + 0.2 - 0.3;
  printf("%.17g\n", x);  // May not be exactly zero across platforms
  ```

#### 2. ⚙️ **Compiler Optimizations**

* Flags like `-O2`, `-ffast-math`, or `-mfma` can reorder or fuse operations.
* `-ffast-math` is **not** reproducible—it breaks strict IEEE 754 compliance.

#### 3. 🧵 **Multithreading and Race Conditions**

* Threads may execute in a different order each run.
* Accumulations (e.g., summing arrays) can vary in precision depending on the order of operations.

#### 4. 🕒 **Timestamps, PIDs, and Entropy**

* Code that reads from `time()`, `getpid()`, `/dev/urandom`, etc. will be inherently non-reproducible unless mocked or fixed.

#### 5. 🛠️ **Platform/Hardware Differences**

* Different CPUs (Intel vs AMD) may handle floating point differently.
* Endianness and alignment can affect memory and binary layout.

---

### ✅ **Strategies to Improve Bit Reproducibility in C**

| Challenge                      | Solution                                                                                                    |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------- |
| Floating-point inconsistencies | Use `-ffloat-store`, disable FMA: `-fno-fma`, use double consistently                                       |
| Compiler variability           | Pin compiler version, avoid `-ffast-math`, use `-fno-unsafe-math-optimizations`                             |
| Thread nondeterminism          | Use single-threaded code or deterministic threading libraries                                               |
| Randomness                     | Replace with fixed seed RNG or mock values                                                                  |
| Timestamps                     | Use environment variable like `SOURCE_DATE_EPOCH`, or compile-time macros like `__DATE__` should be avoided |
| Platform variation             | Use same architecture, compiler, and OS; avoid architecture-specific code unless abstracted cleanly         |

---

### 🧪 Example: Bit-Reproducible Matrix Multiply (C, GCC)

```c
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
```

Compile with:

```bash
gcc -O2 -fno-unsafe-math-optimizations -fno-finite-math-only -ffloat-store -march=x86-64 -o matmul matmul.c
```

---

Here’s a **Makefile and test script** setup for testing **bit reproducibility in C**:

---

### 📄 **1. `Makefile`**

```make
CC = gcc
CFLAGS = -O2 -std=c11 -Wall -Wextra \
         -fno-unsafe-math-optimizations -fno-finite-math-only \
         -ffloat-store -march=x86-64

TARGET = matmul
SRC = matmul.c

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $^

test: $(TARGET)
	./$(TARGET) > output.txt
	@# If a reference exists, compare
	@if [ -f reference_output.txt ]; then \
		cmp --silent output.txt reference_output.txt && \
		echo "✅ Output matches reference (bit-for-bit)." || \
		echo "❌ Output differs from reference."; \
	else \
		cp output.txt reference_output.txt && \
		echo "🔖 Reference output saved."; \
	fi

clean:
	rm -f $(TARGET) output.txt reference_output.txt
```

---

### 🧪 **2. `matmul.c` (Same as Before)**

Use the same `matmul.c` file from the previous message. It's a deterministic 2×2 matrix multiplication written in plain C.

---

### 🧼 **3. Run Instructions**

```bash
make          # Compile with strict reproducibility flags
make test     # Run and compare against reference (bitwise)
```

* First run creates `reference_output.txt`.
* Future runs compare `output.txt` against it using `cmp`.

---

### 📦 Optional: Run Across Systems

* Run on both AMD and Intel systems.
* Make sure same:

  * Compiler version (e.g. GCC 12.3)
  * `make` settings
  * Linux distro (or use Docker to ensure parity)

---

# AOCL‑BLAS and Cross‑CPU Consistency 
AOCL-BLAS is bit-reproducible across different AMD hardware configurations implies checking if BLAS operations using AOCL-BLAS produce identical binary outputs on various AMD CPUs, under the same software and build settings.
AMD’s AOCL‑BLAS is a tuned BLIS-based library for Zen CPUs, dynamically selecting CPU‑specific kernels at run time.  By default it chooses the “best” kernel for each processor (e.g. a Zen4-optimized GEMM on Genoa vs a Zen3 one on Milan), so identical code can produce slightly different floating‑point results on different AMD chips. In fact, note that *bitwise* reproducibility holds only if the hardware (CPU microarchitecture, instruction set, number of threads, etc.) is unchanged. Changing the CPU family or BLAS backend typically leads to small round‑off differences.  Similarly, multithreaded execution can reorder summations or use atomic reductions, which can break bit-for-bit identity across runs.  In short, AOCL‑BLAS **does not guarantee** identical  results on EPYC vs Ryzen under normal operation.

However, AOCL‑BLAS does provide knobs to enforce a uniform code path.  For example, setting the environment variable `BLIS_ARCH_TYPE` (to one of `{zen4, zen3, zen2, zen, generic}`) will “completely override” the automatic dispatch.  This forces the library to use a single architecture’s kernels (for instance, specifying `zen3` on both platforms).  In practice, you can pick the lowest common denominator (e.g. `zen2` or even `generic`) so that both CPUs run the *same* code path.  Likewise, `BLIS_MODEL_TYPE={Milan, Genoa, …}` can be used to pick a particular processor model (Milan/Milan‑X=Zen3, Genoa/Bergamo=Zen4).  Using these variables in concert can yield *nearly* identical results: Setting a fixed model/arch “may provide consistent results across different models if consistency is a higher priority than best performance”.  (Note that forcing an incompatible code path will cause errors – e.g. Zen4 kernels on a Zen3 CPU can cause illegal-instruct.)

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

Following section covers reproducibility test scripts for the AOCL-BLAS functions `dgemm`, `daxpy`, and `dot`, focused on ensuring bit-for-bit identical results across AMD systems.

The test suite will include examples, comparison logic, and configuration settings to maximize reproducibility.


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
---
### **Guide to Making Python/Numpy code more floating-point stable and reproducible**
---
*Python/Numpy*, is widely used and often suffers from non-reproducibility in scientific computing or ML:
## **1. Fix All Random Seeds**

Set the seed in all relevant libraries:

```python
import random
import numpy as np

random.seed(42)
np.random.seed(42)
```

If you use **PyTorch** or **TensorFlow**, you'll also need to set their seeds separately.

---

## **2. Avoid Accumulating Floating-Point Errors**

Use **stable algorithms** and **higher precision** where necessary:

### Bad (prone to rounding error):

```python
x = np.linspace(0.1, 1e8, 100000)
y = np.sum(x - x)
```

### Better:

```python
y = np.sum(x - x, dtype=np.float64)
```

Or use `np.dot`/`np.einsum` or `np.cumsum` when possible — they are more stable than manual summation.

---

## **3. Use `np.allclose()` or `np.isclose()` When Comparing Floats**

Avoid `==` with floats — instead:

```python
np.allclose(a, b, rtol=1e-12, atol=1e-15)
```

This tolerates tiny differences due to rounding.

---

## **4. Use Deterministic BLAS/LAPACK Backends**

If you're doing linear algebra (e.g., matrix multiplication, `np.linalg.solve`), backend libraries like **AOCL-BLAS**, **OpenBLAS** or **MKL** may use parallelism or reordering, leading to slight variation.

* **Set environment variables** to control behavior:

```bash
export OMP_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

This forces deterministic single-threaded behavior.

---

## **5. Turn Off Numpy's Implicit Multithreading (if needed)**

To ensure reproducibility:

```python
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
```

---

## **6. Use Decimal for Exact Decimal Arithmetic**

For critical calculations:

```python
from decimal import Decimal, getcontext

getcontext().prec = 50  # set precision
a = Decimal('0.1')
b = Decimal('0.2')
c = a + b
print(c)  # will be exactly 0.3
```

Slower, but useful in finance or high-precision scenarios.

---
Here’s a **Jupyter-ready Python example** that compares `float32`, `float64`, and `Decimal` arithmetic to show how floating-point precision affects results:

---

### **Example: Comparing 0.1 + 0.2**

```python
import numpy as np
from decimal import Decimal, getcontext

# Set precision for Decimal
getcontext().prec = 50

# float32
a32 = np.float32(0.1)
b32 = np.float32(0.2)
c32 = a32 + b32
print("float32: 0.1 + 0.2 =", c32)

# float64 (default Python float)
a64 = 0.1
b64 = 0.2
c64 = a64 + b64
print("float64: 0.1 + 0.2 =", c64)

# Decimal
aD = Decimal('0.1')
bD = Decimal('0.2')
cD = aD + bD
print("Decimal: 0.1 + 0.2 =", cD)

# Comparison to 0.3
print("\nComparison to 0.3:")
print("float32 == 0.3?", c32 == np.float32(0.3))
print("float64 == 0.3?", c64 == 0.3)
print("Decimal == 0.3?", cD == Decimal('0.3'))
```

---

### **Expected Output:**

You’ll likely see:

```
float32: 0.1 + 0.2 = 0.30000001192092896
float64: 0.1 + 0.2 = 0.30000000000000004
Decimal: 0.1 + 0.2 = 0.3

Comparison to 0.3:
float32 == 0.3? False
float64 == 0.3? False
Decimal == 0.3? True
```

---

### **Takeaways:**

* `float32` and `float64` can’t exactly represent numbers like 0.1 or 0.2.
* `Decimal` gives exact decimal results (but is much slower).
* Always use `np.allclose` or `Decimal` if exact comparison is needed.

Here’s a **Jupyter-ready Python example** showing how floating-point precision affects the accuracy of large summations — a classic case where **error accumulation** becomes visible:

---

### **Example: Summing 1 million small numbers**

```python
import numpy as np
from decimal import Decimal, getcontext

# Set precision for Decimal
getcontext().prec = 50

# Create 1 million small numbers
n = 1_000_000
value = 1e-6

# float32 sum
arr32 = np.full(n, value, dtype=np.float32)
sum32 = np.sum(arr32, dtype=np.float32)
print("float32 sum:", sum32)

# float64 sum
arr64 = np.full(n, value, dtype=np.float64)
sum64 = np.sum(arr64, dtype=np.float64)
print("float64 sum:", sum64)

# Decimal sum
arrD = [Decimal(str(value))] * n
sumD = sum(arrD)
print("Decimal sum:", sumD)

# Expected exact value
exact = Decimal(str(value)) * n
print("\nExpected sum:", exact)

# Differences
print("\nDifference from expected:")
print("float32 error:", abs(Decimal(str(sum32)) - exact))
print("float64 error:", abs(Decimal(str(sum64)) - exact))
```

---

### **Expected Output (Approximate):**

```
float32 sum: 0.99999994
float64 sum: 1.0
Decimal sum: 1.000000

Expected sum: 1.000000

Difference from expected:
float32 error: 6.000000e-08
float64 error: 0E-50
```

---

### **Takeaways:**

* **float32** accumulates significant rounding error when summing many small values.
* **float64** is much more accurate, but still not bit-perfect in all cases.
* **Decimal** gives perfect results (but is slow and memory-heavy).

This kind of difference can grow much worse in scientific simulations, machine learning training loops, or financial calculations.

Testing **bit reproducibility** of a workload means checking whether repeated executions under *identical conditions* produce *exactly the same binary output* — not just numerically close, but **bit-for-bit identical**.

Here’s a structured approach:

---

## **Step-by-Step: How to Test Bit Reproducibility**

### **1. Run the Workload Twice**

Run your program twice (or more) under the *same environment*, using scripts or automation to ensure nothing changes.

Example (Linux shell):

```bash
./run_simulation > output_run1.bin
./run_simulation > output_run2.bin
```

Or with random seed controlled:

```bash
./run_simulation --seed 42 > output_run1.bin
./run_simulation --seed 42 > output_run2.bin
```

---

### **2. Compare Outputs Bit-for-Bit**

Use `diff`, `cmp`, `sha256sum`, or Python for bytewise comparison.

#### Shell:

```bash
cmp output_run1.bin output_run2.bin
# or
sha256sum output_run1.bin output_run2.bin
```

#### Python:

```python
with open("output_run1.bin", "rb") as f1, open("output_run2.bin", "rb") as f2:
    identical = f1.read() == f2.read()
print("Bitwise identical?" , identical)
```

If you’re producing textual or CSV outputs, normalize formatting first (e.g., remove timestamps, sort rows).

---

### **3. Use Checksums for Large Files**

For large files, hashing is faster:

```bash
sha256sum output_run*.bin
```

Identical checksums mean the outputs are **bitwise identical**.

---

### **4. Automate It in CI or a Script**

Include this reproducibility test in your workflow:

```bash
#!/bin/bash
./your_workload --seed 123 --threads 1 > run1.out
./your_workload --seed 123 --threads 1 > run2.out

if cmp -s run1.out run2.out; then
    echo "Bit reproducible!"
else
    echo "Not reproducible!"
    exit 1
fi
```

---

### **5. Check Environmental Variables**

Ensure environmental factors are fixed:

* Threads: `OMP_NUM_THREADS=1`
* CPU affinity (optional): `taskset` or `numactl`
* Locale: `export LC_ALL=C`
* Math: `-fno-fast-math`, fixed seeds

---

## **Bonus: Tools That Help**

* **ReproZip**: Captures all dependencies to reproduce runs elsewhere.
* **Guix/Nix**: Reproducible software environments.
* **Git LFS**: Tracks binary outputs in ML workflows for reproducibility auditing.

---
Here’s a **ready-made Python script** that tests **bit reproducibility** for any function or model by comparing binary dumps of its outputs:

---

### **Python Script: `test_bit_reproducibility.py`**

```python
import numpy as np
import random
import pickle
import hashlib

def hash_bytes(obj):
    """Serialize and hash a Python object using SHA-256"""
    b = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    return hashlib.sha256(b).hexdigest()

def run_workload(seed=42):
    """Example workload: replace with your actual function"""
    random.seed(seed)
    np.random.seed(seed)

    # Replace this with your real computation
    data = {
        "array": np.random.rand(1000),
        "value": sum(np.random.randint(0, 100, 100))
    }
    return data

def test_bit_reproducibility():
    result1 = run_workload(seed=42)
    result2 = run_workload(seed=42)

    hash1 = hash_bytes(result1)
    hash2 = hash_bytes(result2)

    print("Hash 1:", hash1)
    print("Hash 2:", hash2)

    if hash1 == hash2:
        print("Result: Bit reproducible!")
    else:
        print("Result: Not bit reproducible!")

if __name__ == "__main__":
    test_bit_reproducibility()
```

---

### **How It Works:**

* Uses `pickle` to serialize Python objects (arrays, dicts, numbers).
* Hashes the serialized bytes with `SHA-256`.
* Compares the hashes to detect *any* bit-level difference.

---

### **To Adapt This:**

Replace the `run_workload()` function with **your model**, simulation, or data processing code — just make sure it returns a consistent Python object like `dict`, `list`, `np.ndarray`, etc.

---
Here’s an enhanced version of the reproducibility tester that can run your workload in **separate processes**, optionally **on different machines or platforms**, and test whether the results are **bit-for-bit identical**.

---

## ✅ Bit-Reproducibility Tester Across Processes (Portable & Cross-Platform)

### **`test_reproducibility_multiprocess.py`**

```python
import numpy as np
import random
import pickle
import hashlib
import subprocess
import tempfile
import os
import sys

def run_workload(seed=42):
    """Your reproducible workload goes here."""
    random.seed(seed)
    np.random.seed(seed)

    # Replace this logic with your own computation
    result = {
        "array": np.random.rand(1000),
        "value": np.sum(np.random.randint(0, 100, 100))
    }
    return result

def save_output(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def hash_output(path):
    with open(path, 'rb') as f:
        data = f.read()
        return hashlib.sha256(data).hexdigest()

def child_process(output_path, seed):
    result = run_workload(seed=seed)
    save_output(result, output_path)

def test_cross_process_reproducibility():
    with tempfile.TemporaryDirectory() as tmpdir:
        path1 = os.path.join(tmpdir, "output1.pkl")
        path2 = os.path.join(tmpdir, "output2.pkl")

        # Run two separate processes
        subprocess.run([sys.executable, __file__, "--child", path1, "42"], check=True)
        subprocess.run([sys.executable, __file__, "--child", path2, "42"], check=True)

        # Compare hashes
        hash1 = hash_output(path1)
        hash2 = hash_output(path2)

        print("Hash 1:", hash1)
        print("Hash 2:", hash2)

        if hash1 == hash2:
            print("✅ Bit reproducible across processes!")
        else:
            print("❌ Not bit reproducible!")

if __name__ == "__main__":
    if "--child" in sys.argv:
        _, _, path, seed = sys.argv
        child_process(path, int(seed))
    else:
        test_cross_process_reproducibility()
```

---

## 💡 How to Use

### **Run it like this:**

```bash
python test_reproducibility_multiprocess.py
```

### **What It Does:**

* Spawns **two child processes**.
* Each runs the same function with the same seed and writes output to disk.
* Hashes both outputs and checks for **bit-level identity**.

---

## 🧪 To Test Across Platforms

1. Save `run_workload()` and `child_process()` to a standalone script.
2. Copy to multiple platforms (e.g. Intel vs AMD, Docker vs native).
3. Run on both and compare output files using:

```bash
sha256sum output1.pkl output2.pkl
# or
cmp output1.pkl output2.pkl
```

---
Below is an extended version of the **bit-reproducibility tester using MPI**, via the popular [**mpi4py**](https://mpi4py.readthedocs.io/) Python library. This allows you to check whether your distributed computation produces **bitwise identical results** across **MPI ranks**, which is especially useful for scientific simulations or parallel numerical workloads.

---

## ✅ Bit-Reproducibility Test with MPI (using `mpi4py`)

### **Requirements**

Install `mpi4py` if you haven’t:

```bash
pip install mpi4py
```

Ensure you have an MPI implementation installed (`OpenMPI`, `MPICH`, etc.).

---

### **`mpi_bit_repro_test.py`**

```python
from mpi4py import MPI
import numpy as np
import random
import hashlib
import pickle
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

SEED = 42  # Global fixed seed

def run_workload(seed=SEED, rank=0):
    random.seed(seed + rank)  # make seed vary with rank if needed
    np.random.seed(seed + rank)

    # Simulate a computation — replace with your real workload
    result = {
        "rank": rank,
        "data": np.random.rand(1000),
        "value": np.sum(np.random.randint(0, 100, 100))
    }
    return result

def hash_result(obj):
    b = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    return hashlib.sha256(b).hexdigest()

def main():
    result = run_workload(seed=SEED, rank=rank)
    result_hash = hash_result(result)

    # Gather all hashes to rank 0
    all_hashes = comm.gather(result_hash, root=0)

    if rank == 0:
        print("=== Bitwise Reproducibility Report ===")
        reference = all_hashes[0]
        all_identical = all(h == reference for h in all_hashes)
        for i, h in enumerate(all_hashes):
            print(f"Rank {i}: {h}")
        if all_identical:
            print("✅ All ranks produced bitwise identical results!")
        else:
            print("❌ Bitwise differences detected between ranks!")

if __name__ == "__main__":
    main()
```

---

### **Run with MPI**

```bash
mpirun -n 4 python mpi_bit_repro_test.py
```

You’ll see SHA-256 hashes printed per rank, followed by a reproducibility verdict.

---

## ⚙️ Options for More Control

You can modify the seed behavior depending on whether:

* You **expect all ranks to be identical** (use same seed: `SEED`)
* You **expect reproducibility across runs but not between ranks** (e.g. use `seed + rank` but compare each rank across runs)

To **compare multiple MPI runs**, save each rank’s output to file and hash it separately later.

---

## 🔒 Ensuring MPI Reproducibility

To maximize bit-reproducibility in MPI jobs:

* Use **deterministic communication order** (avoid race conditions in `reduce`, `gather`, etc.)
* Set a **fixed seed per rank**
* Use **single-threaded math backends**:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

---
Here's a version of the **MPI-based bit reproducibility test** extended to support **multi-machine environments** and **audit logging**. It logs hash results per rank into a shared log file, so you can trace and debug any mismatches.

---

## ✅ Distributed Bit-Reproducibility Checker with Logging (for MPI clusters)

### **`mpi_bit_repro_audit.py`**

```python
from mpi4py import MPI
import numpy as np
import random
import hashlib
import pickle
import os
import socket
from datetime import datetime

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

SEED = 42
LOG_FILE = "bit_repro_audit.log"

def run_workload(seed=SEED, rank=0):
    random.seed(seed + rank)
    np.random.seed(seed + rank)

    # Example: Replace with your real distributed computation
    result = {
        "rank": rank,
        "host": socket.gethostname(),
        "data": np.random.rand(1000),
        "value": np.sum(np.random.randint(0, 100, 100))
    }
    return result

def hash_result(obj):
    b = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    return hashlib.sha256(b).hexdigest()

def log_to_file(log_entry):
    """Appends a log entry to a shared log file (rank 0 only)."""
    with open(LOG_FILE, "a") as f:
        f.write(log_entry + "\n")

def main():
    result = run_workload(seed=SEED, rank=rank)
    result_hash = hash_result(result)

    # Collect all hashes
    all_hashes = comm.gather(result_hash, root=0)
    all_hosts = comm.gather(socket.gethostname(), root=0)

    if rank == 0:
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entries = [f"[{now}] Rank {i} on {host}: {h}"
                       for i, (host, h) in enumerate(zip(all_hosts, all_hashes))]
        for entry in log_entries:
            print(entry)
            log_to_file(entry)

        reference = all_hashes[0]
        all_identical = all(h == reference for h in all_hashes)

        result_line = (
            f"[{now}] ✅ Reproducible across {size} ranks"
            if all_identical else
            f"[{now}] ❌ Bit mismatch detected!"
        )
        print(result_line)
        log_to_file(result_line)
    comm.Barrier()

if __name__ == "__main__":
    main()
```

---

## 🧪 How to Run on a Cluster

This script supports multi-node MPI jobs. Example (for OpenMPI):

```bash
mpirun -np 8 -hostfile my_hosts.txt python mpi_bit_repro_audit.py
```

Where `my_hosts.txt` might contain:

```
node01 slots=4
node02 slots=4
```

---

## 📝 Output Example

```
[2025-05-13 14:26:32] Rank 0 on node01: a1bc...91
[2025-05-13 14:26:32] Rank 1 on node01: a1bc...91
[2025-05-13 14:26:32] Rank 2 on node02: a1bc...91
[2025-05-13 14:26:32] Rank 3 on node02: a1bc...91
[2025-05-13 14:26:32] ✅ Reproducible across 4 ranks
```

Or:

```
[2025-05-13 14:26:35] ❌ Bit mismatch detected!
```

---

## 🔒 Best Practices for Cross-Machine Reproducibility

* Fix RNG seed logic per rank
* Ensure all nodes use the same:

  * Python version
  * NumPy version
  * BLAS/LAPACK library (MKL, AOCL_BLAS, OpenBLAS, etc.)
* Limit threads:

  ```bash
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  ```
* Use deterministic communication patterns in your code

---
## **Conclusion**

Bit reproducibility is critical for scientific rigor, debugging, and security. While achieving exact reproducibility across different CPUs or architectures is difficult, careful control of floating-point behavior, threading, and environment can minimize variability. For strict reproducibility, use deterministic libraries, single-threaded execution, and fixed seeds.


