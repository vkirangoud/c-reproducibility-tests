CC = gcc
CFLAGS = -O2 -std=c11 -Wall -Wextra -ffloat-store -fno-unsafe-math-optimizations
OPENMP = -fopenmp
BIN_DIR = bin
SRC_DIR = examples

PROGRAMS = \
	$(BIN_DIR)/accumulation_order \
	$(BIN_DIR)/x87_precision \
	$(BIN_DIR)/math_reordering \
	$(BIN_DIR)/openmp_sum \
	$(BIN_DIR)/random_seed

all: $(PROGRAMS)

$(BIN_DIR)/%: $(SRC_DIR)/%.c | $(BIN_DIR)
	$(CC) $(CFLAGS) $(if $(findstring openmp,$@),$(OPENMP),) -o $@ $<

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

clean:
	rm -rf $(BIN_DIR) *.out *.ref
	