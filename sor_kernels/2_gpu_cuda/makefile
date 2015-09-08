OBJECTS=cuda_sor

all: $(OBJECTS)

cuda_sor: cuda_sor.cu
	nvcc -O1 -lrt -lm -arch=sm_20 -o $@ $^