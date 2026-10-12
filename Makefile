# Builds rh_worker_cuda for RTX 5070 Ti (Blackwell sm_120, CUDA 13.1)
# Run: make
# Then the coordinator auto-detects and prefers the binary over Python

NVCC    := nvcc
TARGET  := rh_worker_cuda
SRC     := rh_cuda.cu

# sm_120 = Blackwell (RTX 50xx series)
ARCH    := -gencode arch=compute_120,code=sm_120

# -O3 for host code; --use_fast_math only for float — we keep FP64 exact
NVCCFLAGS := $(ARCH) -O3 -std=c++17 \
             -Xcompiler "-O3 -march=native" \
             --extra-device-vectorization \
             -lineinfo

# link math and CUDA runtime; $ORIGIN rpath = look next to the binary for libcufft
CUDA_LIBDIR := /usr/local/cuda-13.0/lib64
LDFLAGS := -lm -lcufft -cudart static -Xlinker '-rpath,$$ORIGIN'

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LDFLAGS)
	@echo "Built: $@"
	@./$@ --version 2>/dev/null || true

clean:
	rm -f $(TARGET)

# Distribute and build on all 4 nodes
install: $(TARGET)
	@cp $(CUDA_LIBDIR)/libcufft.so.12 . 2>/dev/null || true
	@for node in 192.168.100.10 192.168.100.11 192.168.100.12; do \
	    echo "=== $$node ==="; \
	    scp -q $(TARGET) libcufft.so.12 rh_worker.py rh_core.py nas_fs.py $$node:/home/techno/math/ && \
	    ssh $$node "chmod +x /home/techno/math/$(TARGET)" && echo "  deployed"; \
	done

.PHONY: all clean install
