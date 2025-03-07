
# Change these configs
CUDA_PATH ?= /usr/local/cuda
NVCC      := $(CUDA_PATH)/bin/nvcc
ARCH      ?= sm_89

SRCS := $(wildcard *.cu)

EXES := $(patsubst %.cu, %, $(SRCS))

all: $(EXES)

%: %.cu
	$(NVCC) -arch=$(ARCH) -O3 $< -o $@

# Clean up
clean:
	rm -f $(EXES)

.PHONY: all clean
