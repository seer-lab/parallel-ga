NVCC=nvcc
CUDA_INCLUDEPATH=/usr/local/cuda-11.2/include
NVCC_OPTS=-O3 -arch=sm_60 -Xcompiler -Wall -Xcompiler -Wextra -m64


sphere: sphere.o evaluate.o Makefile
	$(NVCC) -o sphere sphere.o evaluate.o -L $(NVCC_OPTS)

sphere.o: sphere.cpp 
	g++ -c sphere.cpp $(GCC_OPTS) -I $(CUDA_INCLUDEPATH) 

evaluate.o: evaluate.cu
	nvcc -c -arch=sm_60 evaluate.cu 

clean:
	rm -f *.o *.png a