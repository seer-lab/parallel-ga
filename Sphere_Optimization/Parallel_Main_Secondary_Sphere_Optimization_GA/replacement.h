#ifndef __replacement_h__
#define __replacement_h__

#include <vector>
#include <cfloat>
#include <cmath> 

int bestIndividual(double *a, int n);

void replacement (double *population, double* temp_population, double *fitness, const int p, const int populationSize, const int elitism);

#endif

