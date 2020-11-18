#include <stdlib.h>

void initPopulation (int *population, const int solutionsPerPopulation, const int length) {

    for(int i = 0; i < solutionsPerPopulation; i++) {
        
        for (int j = 0; j < length; j++) {

            *((population+i*solutionsPerPopulation) + j) = rand() % 2;

        }
    }
}