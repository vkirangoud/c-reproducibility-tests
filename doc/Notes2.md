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
Slight floating-point differences between **Intel** and **AMD** CPUs typically stem from subtle variations in:

---

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

Great — making floating-point code more **consistent and reproducible across platforms** requires carefully controlling precision, math behavior, and compiler settings.

Here are **practical tips** to help:

---

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
Great — let’s go with **Python/Numpy**, since it's widely used and often suffers from non-reproducibility in scientific computing or ML.

Here’s a **guide to making Python/Numpy code more floating-point stable and reproducible**:

---

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

