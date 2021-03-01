#ifndef __mutation_h__
#define __mutation_h__

#include <vector>
#include <stdlib.h>
using std::vector;


void mutation (vector<vector<double>> offsprings, std::vector<float> bounds,
               vector<vector<double>> &mutants, const int p);
               
#endif