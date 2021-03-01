#ifndef __replacement_h__
#define __replacement_h__

#include <vector>
using std::vector;

int find_min(vector<double> a, int n);

void replacement (vector<vector<double>> temp_population, vector<vector<double>> &population,  
                  vector<double> &fitness, const int populationSize, const int p);
#endif

