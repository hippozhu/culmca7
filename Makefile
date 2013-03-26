all: test

test: main.o SVMData.o kernels.o
	g++ -g main.o SVMData.o kernels.o -o test -L/home/yzhu7/.local/cuda/lib64 -lcuda -lcudart

main.o: main.cpp
	g++ -g -c main.cpp

SVMData.o: SVMData.cpp
	g++ -g -c SVMData.cpp

kernels.o: kernels.cu
	nvcc -g -G -c -gencode arch=compute_20,code=sm_20 kernels.cu

clean:
	rm -rf *.o test
