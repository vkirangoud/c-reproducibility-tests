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
