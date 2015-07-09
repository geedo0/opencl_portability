all: convolve

utils.o: utils.c
	gcc -c $^ -o $@ -lrt

convolve: main.c utils.o
	gcc -o $@ $^ `pkg-config --cflags --libs MagickWand` -lrt

clean:
	rm *.o convolve