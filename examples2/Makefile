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
