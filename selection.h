#include <stdio.h> 
#include <stdlib.h>
#include <string.h>

 #define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

void selection (int *fitness, int *parents, int *population, const int length) {

    // generate 4 indexs between 0-9

    int individualPool[4];
    memset(individualPool, 0, 4*sizeof(int));


    for (size_t i = 0; i < 4; i++) {
        individualPool[i] = rand() % length;
    }

    int parentIndex1 = 0;
    int parentIndex2 = 0;
    
    for (size_t i = 0; i < length; i++) {
        
        if (fitness[i] == max(fitness[individualPool[0]], fitness[individualPool[1]]))
            parentIndex1 = i;

        if (fitness[i] == max(fitness[individualPool[2]], fitness[individualPool[3]]))
            parentIndex2 = i;
    }
    
    
    for(int i = 0; i < length; i++) {
        *((parents+0*length) + i) = *((population+parentIndex1*length) + i);
        *((parents+1*length) + i) = *((population+parentIndex2*length) + i);
    }
}