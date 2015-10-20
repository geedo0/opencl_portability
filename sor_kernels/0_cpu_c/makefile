OBJECTS		= sor_omega sor_optimizing sor_threaded
COMMON		= ../common

all: $(OBJECTS)

sor_omega: sor_omega.c $(COMMON)/utils.o
	gcc -lm -O1 -o $@ -I$(COMMON) $^

sor_optimizing: sor_optimizing.c $(COMMON)/utils.o
	gcc -lm -lrt -O1 -o $@ -I$(COMMON) $^

sor_threaded: sor_threaded.c $(COMMON)/utils.o
	gcc -pthread -lm -lrt -O1 -o $@ -I$(COMMON) $^

$(COMMON)/utils.o: $(COMMON)/utils.c
	gcc -c -O1 -o $@ $^

clean:
	rm $(OBJECTS)
	rm $(COMMON)/*.o
