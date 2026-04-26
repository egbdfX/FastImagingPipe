CUDA_HOME ?= /usr/local/cuda

CFITSIO_HOME := $(HOME)/Libraries/cfitsio

INC := -I$(CUDA_HOME)/include -I$(CFITSIO_HOME)/include -I.

LIB := -L$(CUDA_HOME)/lib64 -L$(CFITSIO_HOME)/lib -lcudart -lcfitsio -lcufft

GCC := g++
NVCC := $(CUDA_HOME)/bin/nvcc

GCC_OPTS :=-O3 -fPIC -Wall -Wextra $(INC) -std=c++11
NVCCFLAGS :=-O3 -gencode arch=compute_90,code=sm_90 --ptxas-options=-v -Xcompiler -fPIC -Xcompiler -Wextra -lineinfo $(INC) --use_fast_math

all: clean sharedlibrary_gpu

sharedlibrary_gpu: src/pipe_interface.o src/FIPkernels.o
	$(NVCC) -o sharedlibrary_gpu $(NVCCFLAGS) src/pipe_interface.o src/FIPkernels.o $(LIB)

src/pipe_interface.o: src/pipe_interface.cpp
	$(GCC) -c src/pipe_interface.cpp $(GCC_OPTS) -o src/pipe_interface.o

src/FIPkernels.o: src/FIPkernels.cu
	$(NVCC) -c src/FIPkernels.cu  $(NVCCFLAGS) -o src/FIPkernels.o

clean:
	rm -f src/*.{o,so} *.{o,.so}
