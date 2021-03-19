#ifndef __selection_h__
#define __selection_h__

#include <vector>
#include <stdlib.h>
#include <cfloat>
using std::vector;

int bestIndividual(double *a, std::vector<int> b, int n);

void tournamentSelection(double *parents, double* population, double *fitness, 
                         const int p, const int populationSize, const int tournamentSize);
                
#endif