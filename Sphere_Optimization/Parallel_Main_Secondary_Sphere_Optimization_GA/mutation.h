#ifndef __mutation_h__
#define __mutation_h__

#include <vector>
#include <stdlib.h>
using std::vector;

void mutation(double *temp_population, std::vector<float> bounds, const int p, const int populationSize, const float mutationProbability);
               
#endif