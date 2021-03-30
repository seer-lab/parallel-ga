#include "../include/selection.h"

int bestIndividual(double *a, std::vector<int> b, int n) {
    
    unsigned int i, index = b[0];
    double diff = DBL_MAX;

    for (i = 0; i < n; i++) {
        double absVal = abs(a[b[i]]);
        if (absVal < diff) {
            index = b[i];
            diff = absVal;
        } else if (absVal == diff && a[b[i]] > 0 && a[index] < 0) {
            index = b[i];
        }
    }

    return index;
}

void tournamentSelection(double *parents, double* population, double *fitness, const int p, const int populationSize, const int tournamentSize) {

    std::vector<int> tournamentPool(tournamentSize, 0);

    unsigned int count = 0;
    int parentIndex = 0;
    while (count < populationSize) {
        
        // selecting individuals for tournament
        for (unsigned int i = 0; i != tournamentSize; i++) 
            tournamentPool[i] = rand() % populationSize; 

        parentIndex = bestIndividual(fitness, tournamentPool, tournamentSize);

        
        for (unsigned int i = 0; i < p; i++) 
            parents[count * p + i] = population[parentIndex * p + i];

        count++;
    }

}