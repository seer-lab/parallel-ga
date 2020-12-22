#include <stdio.h> 
#include <stdlib.h>

void crossover (int *parents, int *offsprings, const int length) {

    int crossoverPoint = rand() % length;

    for (size_t i = 0; i < crossoverPoint; i++) {
        *((offsprings+0*length) + i) = *((parents+0*length) + i);
        *((offsprings+1*length) + i) = *((parents+1*length) + i);
    }

    for (size_t i = crossoverPoint; i < length; i++) {
        *((offsprings+0*length) + i) = *((parents+1*length) + i);
        *((offsprings+1*length) + i) = *((parents+0*length) + i);
    }

}