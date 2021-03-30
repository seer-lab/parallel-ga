#include "../include/evaluation.cuh"

# define M_PI           3.14159265358979323846  /* pi */

__device__
double square(double x) { return x * x; }

__device__
void sphere(double *islandFitness, double *islandPop, const int p) {
    for (unsigned int i = 0; i < p; i++)
        islandFitness[threadIdx.x] += square(islandPop[threadIdx.x * p + i]);
}

__device__
void rastrigin(double *islandFitness, double *islandPop, const int p) {
    for (unsigned int i = 0; i < p; i++)
        islandFitness[threadIdx.x] += ((islandPop[threadIdx.x * p + i]*islandPop[threadIdx.x * p + i]) 
                                    - (10*cos(2*M_PI*islandPop[threadIdx.x * p + i])) +10);
}

__device__
void ackley(double *islandFitness, double *islandPop, const int p) {
    double a = 20.0, b = 0.2, c = 2*M_PI;
    double s1 = 0.0, s2 = 0.0;

    for (unsigned int i = 0; i < p; i++) {
        s1 = s1 + pow(islandPop[threadIdx.x * p + i], 2);
        s2 = s2 + cos(c*islandPop[threadIdx.x * p + i]);
    }

    islandFitness[threadIdx.x] = -a * exp ( -b * rsqrt (s1 / double (p))) 
                                - exp (s2 / double (p)) + a + exp(1.0);
}

__device__
void griewank(double *islandFitness, double *islandPop, const int p) {
    double s = 0.0, t = 0.0;

    for (unsigned int i = 0; i < p; i++) {
        s += pow(islandPop[threadIdx.x * p + i], 2);
        t *= cos(islandPop[threadIdx.x * p + i] / rsqrt(double (i+1)));
    }

    islandFitness[threadIdx.x] = (s / 4000.0) - t + 1;
}