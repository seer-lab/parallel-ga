#ifndef __evaluation_h__
#define __evaluation_h__

#include <vector>
using std::vector;

double square(double x);

double sphere(vector<double> individual, const int p);

void evaluation (vector<vector<double>> population, vector<double> &fitness, const int p);

#endif
