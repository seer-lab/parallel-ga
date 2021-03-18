GCC=g++

sphere: evaluation.o mutation.o population.o crossover.o replacement.o selection.o GA.o Makefile
	$(GCC) -o sphere evaluation.o mutation.o population.o crossover.o replacement.o selection.o GA.o

mutation.o: mutation.cpp
	$(GCC) -c mutation.cpp 

population.o: population.cpp 
	$(GCC) -c population.cpp 

crossover.o: crossover.cpp
	$(GCC) -c crossover.cpp 

replacement.o: replacement.cpp 
	$(GCC) -c replacement.cpp 

selection.o: selection.cpp 
	$(GCC) -c selection.cpp 

evaluation.o: evaluation.cpp
	$(GCC) -c evaluation.cpp 

GA.o: GA.cpp
	$(GCC) -c GA.cpp 

clean:
	rm -f *.o *.png a