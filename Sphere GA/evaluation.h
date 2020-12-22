#include <stdio.h> 
#include <vector>

using std::vector;


double square(double x) { return x * x;}

double sphere(vector<double> individual, const int p) { 

    double sum = 0.0;
    for (vector<int>::size_type i = 0; i != p; i++)
        sum += square(individual[i]);

    return sum; 
}

void evaluation (vector<vector<double>> population, vector<double> &fitness, const int p) {

    for (vector<int>::size_type i = 0; i != fitness.size(); i++) 
        fitness[i] = sphere(population[i], p);
} 
