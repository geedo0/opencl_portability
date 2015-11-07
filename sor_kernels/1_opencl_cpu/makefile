CCFLAGS=-O3 -std=gnu99 -ffast-math
LIBS = -lm -lOpenCL -fopenmp -lrt
COMMON		= ../common

all: sor_ocl

$(COMMON)/utils.o: $(COMMON)/utils.c
	gcc -c $^ -o $@ -lrt

sor_ocl: main.c $(COMMON)/utils.o
	gcc $(CCFLAGS) -o $@ -I$(COMMON) $^ $(LIBS) -g

clean:
	rm *.o sor_ocl