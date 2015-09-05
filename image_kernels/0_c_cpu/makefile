all: convolve_image convolve_matrix

utils.o: ../utils.c
	gcc -c $^ -o $@ -lrt -O1

convolve_image: main.c utils.o
	gcc -o $@ $^ -D USE_MAGICK `pkg-config --cflags --libs MagickWand` -lrt -O1

convolve_matrix: main.c utils.o
	gcc -o $@ $^ -lrt -O1

clean:
	rm *.o convolve