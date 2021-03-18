#ifndef __crossover_h__
#define __crossover_h__

#include <stdlib.h>

void one_point_crossover(double *temp_population, double *parents, const int p, const float crossoverProbability, const int mating);

void arithmetic_crossover(double *temp_population, double *parents, const int p, const float crossoverProbability, const int mating, const float alpha);

#endif