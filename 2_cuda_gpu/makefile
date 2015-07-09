CCFLAGS=-O1 -arch=compute_11

LIBS = -lm -lrt -lcudart -L /usr/local/cuda-5-5/lib

all: convolve

convolve: main.cu
	nvcc $(CCFLAGS) -o $@ $^ $(LIBS) 

clean:
	rm convolve
