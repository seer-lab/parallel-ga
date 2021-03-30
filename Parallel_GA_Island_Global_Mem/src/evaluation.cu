#include "../include/evaluation.cuh"


__device__
double square(double x) { return x * x; }

__device__
void sphere(double *fitness, double *population, const int p, int tid) {
    for (unsigned int i = 0; i < p; i++)
        fitness[tid] += square(population[tid * p + i]);
}

__device__
void rastrigin(double *fitness, double *population, const int p, int tid) {
    for (unsigned int i = 0; i < p; i++)
        fitness[tid] += ((population[tid * p + i]*population[tid * p + i]) 
                        - (10*cos(2*M_PI*population[tid * p + i])) +10);
}

__device__
void ackley(double *fitness, double *population, const int p, int tid) {
    double a = 20.0, b = 0.2, c = 2*M_PI;
    double s1 = 0.0, s2 = 0.0;

    for (unsigned int i = 0; i < p; i++) {
        s1 = s1 + pow(population[tid * p + i], 2);
        s2 = s2 + cos(c*population[tid * p + i]);
    }

    fitness[tid] = -a * exp ( -b * rsqrt (s1 / double (p))) 
                    - exp (s2 / double (p)) + a + exp(1.0);
}

__device__
void griewank(double *fitness, double *population, const int p, int tid) {
    double s = 0.0, t = 0.0;

    for (unsigned int i = 0; i < p; i++) {
        s += pow(population[tid * p + i], 2);
        t *= cos(population[tid * p + i] / rsqrt(double (i+1)));
    }

    fitness[tid] = (s / 4000.0) - t + 1;
}