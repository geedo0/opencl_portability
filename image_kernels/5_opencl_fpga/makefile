CCFLAGS=-O3 -std=gnu99 -ffast-math

LIBS = -lm -lOpenCL -fopenmp -lrt

BOARD_NAME="ProceV_D8"

all: convolve conv.aocx

utils.o: ../utils.c
	gcc -c $^ -o $@ -lrt

convolve: main.c utils.o
	gcc $(CCFLAGS) -o $@ $^  $(LIBS)
	
convolve_image: main.c utils.o
	gcc $(CCFLAGS) -o $@ $^  $(LIBS) `pkg-config --cflags --libs MagickWand` -D USE_MAGICK 

#Intermediate Compilation - Compile time O(minutes)
conv.aoco: conv.cl
	aoc -c -g $^ --report --board ${BOARD_NAME}

#Compile for emulator
conv_emulated.aocx: conv.cl
	aoc -o $@ -march=emulator -g --board ${BOARD_NAME} $^

#Compile for profiler Compile time O(hours)
conv_profiled.aocx: conv.cl
	aoc -o $@ --profile --board $(BOARD_NAME) $^

#Final kernel O(hours)
conv.aocx: conv.cl
	aoc -o $@ --board $(BOARD_NAME) $^

clean:
	rm *.o convolve convolve_image conv.aoco
