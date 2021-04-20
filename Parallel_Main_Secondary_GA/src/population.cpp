#include "../include/population.h"

void initPopulation(double* population, float* bounds, const int col, const int row) {

    for (unsigned int i = 0; i < col; i++)
        for (unsigned int j = 0; j < row; j++)
            population[i*row+j] = bounds[0] + ((double)rand() / RAND_MAX) * (bounds[1] - bounds[0]);

}