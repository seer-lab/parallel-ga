#include "../include/crossover.h"

void one_point_crossover(double *temp_population, double *parents, const int p, const float crossoverProbability, const int mating) {
    int crossoverPoint = 0;
    for (unsigned int i = 0; i < mating; i++) {
        if (((float)rand())/RAND_MAX < crossoverProbability) {
            crossoverPoint = rand() % p;

            for (unsigned int j = 0; j < crossoverPoint; j++) {
                temp_population[i * p + j] = parents[i * p + j];
                temp_population[(i+mating) * p + j] = parents[(i+mating) * p + j];
            }

            for (unsigned int j = crossoverPoint; j < p; j++) {
                temp_population[i * p + j] = parents[(i+mating) * p + j];
                temp_population[(i+mating) * p + j] = parents[i * p + j];
            }
        } else {
            for (unsigned int j = 0; j < p; j++) {
                temp_population[i * p + j] = parents[i * p + j];
                temp_population[(i+mating) * p + j] = parents[(i+mating) * p + j];
            }
        }
    }
}

void arithmetic_crossover(double *temp_population, double *parents, const int p, const float crossoverProbability, const int mating, const float alpha) {
    for (unsigned int i = 0; i < mating; i++) {
        if (((float)rand())/RAND_MAX < crossoverProbability) {
            for(unsigned int j = 0; j < p; j++) {
                temp_population[i * p + j] = alpha * parents[i * p + j] + (1-alpha) * parents[(i+mating) * p + j];
                temp_population[(i+mating) * p + j] = (1-alpha) * parents[i * p + j] + alpha * parents[(i+mating) * p + j];
            }
        } else {
            for (unsigned int j = 0; j < p; j++) {
                temp_population[i * p + j] = parents[i * p + j];
                temp_population[(i+mating) * p + j] = parents[(i+mating) * p + j];
            }  
        }
    }
}