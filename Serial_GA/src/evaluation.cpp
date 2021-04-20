#include "../include/evaluation.h"
#include "../../constants_serial.h"

double square(double x) { return x * x; }

// bounds xi ∈ [-5.12, 5.12]
double sphere(double* individual, int index, const int p) { 
    double sum = 0.0;

    for (unsigned int i = 0; i != p; i++)
        sum += square(individual[index * p + i]);

    return sum; 
}

// bounds xi ∈ [-5.12, 5.12]
double rastrigin(double* individual, int index, const int p) {
    double sum = 0.0;

    for (unsigned int i = 0; i != p; i++)
        sum += ((individual[index * p + i]*individual[index * p + i]) - (10*cos(2*M_PI*individual[index * p + i])) +10);

    return sum; 
}

// bounds xi ∈ [-32.768, 32.768]
double ackley(double* individual, int index, const int p) {
    double sum = 0.0;
    double a = 20.0, b = 0.2, c = 2*M_PI;
    double s1 = 0.0, s2 = 0.0;

    for (unsigned int i = 0; i != p; i++) {
        s1 = s1 + pow(individual[index * p + i], 2);
        s2 = s2 + cos(c*individual[index * p + i]);
    }

    sum = -a * exp ( -b * sqrt (s1 / double (p))) - exp (s2 / double (p)) + a + exp(1.0);

    return sum; 
}

// bounds xi ∈ [-600, 600]
double griewank(double* individual, int index, const int p) {
    double sum = 0.0;
    double s = 0.0, t = 1.0;

    for (unsigned int i = 0; i < p; i++) {
        s += pow(individual[index * p + i], 2);
        t *= cos(individual[index * p + i] / sqrt(i+1));
    }

    sum = (s / 4000.0) - t + 1;

    return sum; 
}

void evaluation(double* population, double* fitness, const int populationSize, const int p) {
    for (unsigned int i = 0; i != populationSize; i++) {
        if (evaluation_type == 1) 
            fitness[i] = sphere(population, i, p);
        else if (evaluation_type == 2)
            fitness[i] = rastrigin(population, i, p);
        else if (evaluation_type == 3)
            fitness[i] = ackley(population, i, p);
        else
            fitness[i] = griewank(population, i, p);
    }
}