#include "replacement.h"

int bestIndividual(double *a, int n) {

    unsigned int i, index = 0;
    double diff = DBL_MAX;
    
    for (i = 0; i < n; i++) {
        double absVal = abs(a[i]);
        if (absVal < diff) {
            index = i;
            diff = absVal;
        } else if (absVal == diff && a[i] > 0 && a[index] < 0) {
            index = i;
        }
    }

    return index;
}

void replacement (double *population, double* temp_population, double *fitness, const int p, const int populationSize, const int elitism) {

    std::vector<int> minIndex(elitism, 0);


    for(unsigned int i = 0; i < elitism; i++) {
        minIndex[i] = bestIndividual(fitness, populationSize);
        fitness[minIndex[i]] = DBL_MAX;
    }
    
    for(unsigned int i = 0; i < elitism; i++)
        for(unsigned int j = 0; j < p; j++)
            population[i * p + j] = population[minIndex[i] * p + j];

    for(unsigned int i = elitism; i < populationSize; i++)
        for(unsigned int j = 0; j < p; j++)
            population[i * p + j] = temp_population[i * p + j];

}