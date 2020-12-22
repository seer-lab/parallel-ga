#include <stdio.h> 
#include <stdlib.h>
#include <string.h>
#include "evaluation.h"

void replacement (int *population,  int *mutants, int *parents, int *fitness, int *itemWeight, int *itemProfit, const int length, const int maxWeight) {

    int lowestIndex[4];
    memset(lowestIndex, 0, 4*sizeof(int));

    for (size_t i = 0; i < 4; i++) {

        lowestIndex[i] = find_min(fitness, length);

        for (size_t j = 0; j < length; j++) {
          
            if (i < 2)
                *((population+lowestIndex[i]*length) + j) = *((parents+(i%2)*length) + j);
            else
                *((population+lowestIndex[i]*length) + j) = *((mutants+(i%2)*length) + j);

        }

        eval(population, fitness, itemWeight, itemProfit, lowestIndex[i], length, maxWeight);   
    }

}