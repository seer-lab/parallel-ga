#include "mutation.h"

void mutation(double *temp_population, std::vector<float> bounds, const int p, const int populationSize, const float mutationProbability) {

    for(unsigned int i = 0; i < populationSize; i++) {
        for(unsigned int j = 0; j < p; j++) {
            if (((float)rand())/RAND_MAX < mutationProbability) 
                temp_population[i * p + j] = bounds[0] + ((double)rand() / RAND_MAX) * (bounds[1] - bounds[0]);
        }
    }

}