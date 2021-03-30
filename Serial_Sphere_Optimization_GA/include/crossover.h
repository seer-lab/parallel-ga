#ifndef __crossover_h__
#define __crossover_h__

#include <stdlib.h>
#include <math.h>

void one_point_crossover(double *temp_population, double *parents, const int p, const float crossoverProbability, const int mating);

void arithmetic_crossover(double *temp_population, double *parents, const int p, const float crossoverProbability, const int mating, const float alpha);

double calc_B(float u, int nc);

void simulated_binary_crossover(double *temp_population, double *parents, const int p, const float crossoverProbability, const int mating, const int nc);

#endif