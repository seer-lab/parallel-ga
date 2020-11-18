#include <stdio.h> 
#include <stdlib.h>

void mutation (int *offsprings, int *mutants, const int length) {

    const float mutationRate = 0.5f;

    for(int i = 0; i < 2; i++) {
        for (int j = 0; j < length; j++) {
            *((mutants+i*length) + j) = *((offsprings+i*length) + j);
        }
    }

    for (size_t i = 0; i < 2; i++) {

        float randomVal = ((float)rand())/RAND_MAX;

        if (randomVal < mutationRate) {
            int randomIndex = rand() % length;

            if (*((mutants+i*length) + randomIndex) == 0) 
                *((mutants+i*length) + randomIndex) = 1;
            else 
                *((mutants+i*length) + randomIndex) = 0; 

        }
    }
}