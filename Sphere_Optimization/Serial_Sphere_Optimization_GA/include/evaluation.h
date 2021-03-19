#ifndef __evaluation_h__
#define __evaluation_h__

#include <vector>
using std::vector;

double square(double x);

double sphere(double* individual, int index, const int p);

void evaluation(double* population, double* fitness, const int populationSize, const int p);

#endif
