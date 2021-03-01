#ifndef __selection_h__
#define __selection_h__

#include <vector>
#include <stdlib.h>
using std::vector;

 #define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })

void selection (vector<vector<double>> &parents, vector<double> fitness,
                vector<vector<double>> population, const int populationSize, const int p);
                
#endif