#ifndef __crossover_h__
#define __crossover_h__

#include <vector>
#include <stdlib.h>

void crossover(double *temp_population, double *parents, const int p, const float crossoverProbability, const int mating);

#endif