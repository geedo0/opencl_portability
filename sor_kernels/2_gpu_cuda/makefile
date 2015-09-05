OBJECTS=cuda_test cuda_sor
CC=nvcc

all: $(OBJECTS)

cuda_test: cuda_test.cu
	$(CC) -o $@ $^

cuda_sor: cuda_sor.cu
	$(CC) -O1 -lrt -lm -arch=sm_20 -o $@ $^