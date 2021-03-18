NVCC=nvcc
GCC=g++

sphere: evaluation.o mutation.o population.o crossover.o replacement.o selection.o GA.o Makefile
	$(NVCC) -arch=sm_60 -o sphere evaluation.o mutation.o population.o crossover.o replacement.o selection.o GA.o

mutation.o: mutation.cpp
	g++ -c mutation.cpp 

population.o: population.cpp 
	g++ -c population.cpp 

crossover.o: crossover.cpp
	g++ -c crossover.cpp 

replacement.o: replacement.cpp 
	g++ -c replacement.cpp 

selection.o: selection.cpp 
	g++ -c selection.cpp 

GA.o: GA.cu
	$(NVCC) -c -arch=sm_60 GA.cu 

evaluation.o: evaluation.cu
	$(NVCC) -c -arch=sm_60 evaluation.cu 

clean:
	rm -f *.o *.png a