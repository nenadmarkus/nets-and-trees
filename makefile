INCLUDE = -I. -I/usr/local/include -I$(HOME)/torch/install/include
LIBOPTS = -L/usr/local/lib/lua/5.1 -L/usr/local/lib -L$(HOME)/torch/install/lib -L$(HOME)/torch/install/lib/lua/5.1
FLAGS = -O3 -fopenmp -lm -fpic -shared

build:
	$(CC) tree.c $(INCLUDE) $(LIBOPTS) $(FLAGS) -o tree.so
