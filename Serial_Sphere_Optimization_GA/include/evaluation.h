#ifndef __evaluation_h__
#define __evaluation_h__

#include <math.h>
#include <iostream>

double square(double x);

double sphere(double* individual, int index, const int p);

double rastrigin(double* individual, int index, const int p);

double ackley(double* individual, int index, const int p);

double griewank(double* individual, int index, const int p);

void evaluation(double* population, double* fitness, const int populationSize, const int p);

#endif
