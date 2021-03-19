#include "../include/evaluation.h"

double square(double x) { return x * x; }

double sphere(double* individual, int index, const int p) { 

    double sum = 0.0;
    for (std::vector<int>::size_type i = 0; i != p; i++)
        sum += square(individual[index * p + i]);

    return sum; 
}

void evaluation(double* population, double* fitness, const int populationSize, const int p) {

    for (unsigned int i = 0; i != populationSize; i++) 
        fitness[i] = sphere(population, i, p);

}