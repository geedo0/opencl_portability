CCFLAGS=-O3 -std=gnu99 -ffast-math

LIBS = -lm -lOpenCL -fopenmp -lrt

all: convolve 

utils.o: utils.c
	gcc -c $^ -o $@ -lrt -g

convolve: main.c utils.o
	cc $(CCFLAGS) -o $@ $^ $(LIBS)

clean:
	rm *.o convolve
