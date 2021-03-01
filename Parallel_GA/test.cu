#include <stdio.h> 
#include <stdlib.h>
#include <cmath> 
#include <cstring>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <cfloat>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>


using std::vector;
using std::cout;

// Generates random values between -5.12 and 5.12
void initPopulation (std::vector<std::vector<double>> &population, std::vector<float> bounds) {

    for (auto i = population.begin(); i != population.end(); i++)
        for (auto j = i->begin(); j != i->end(); j++) 
            *j = bounds[0] + ((double)rand() / RAND_MAX) * (bounds[1] - bounds[0]);
}


void print2dvec(vector<vector<double>> vec) {

    for(auto& i : vec) {
        cout << "[ ";
        for(auto& j : i)
            cout << j << " ";
        cout << "]\n";
    }

}


// square<T> computes the square of a number f(x) -> x*x
template <typename T>
struct square
{
    __host__ __device__
        T operator()(const T& x) const { 
            return x * x;
        }
};

void printvec(vector<double> vec) {

    for(auto& i : vec)
        cout << i << "\n";
}


int main() {   

    
    vector<float> bounds{-5.12, 5.12};

    const int p = 30; // # of genes per individual
    const int populationSize = 100; 

    // Initialize Population 
    vector<vector<double>> population(populationSize, vector<double>(p, 0));
    initPopulation(population, bounds);

    // Evaluating Initial Population
    vector<double> fitness(populationSize, 0);

    // setup arguments
    square<double>        unary_op;
    thrust::plus<double> binary_op;
    double init = 0;
    thrust::device_vector<double> d_population;

    for (std::vector<unsigned int>::size_type i = 0; i < populationSize; i++) {

        d_population = population[i];

        fitness[i] = thrust::transform_reduce(d_population.begin(), d_population.end(), unary_op, init, binary_op);

    }

    printvec(fitness);

    return 0;
    

    /*
    // initialize host array
    double x[4] = {1.0, 2.0, 3.0, 4.0};

    // transfer to device
    thrust::device_vector<double> d_x(x, x + 4);

    // setup arguments
    square<double>        unary_op;
    thrust::plus<double> binary_op;
    double init = 0;

    // compute norm
    double norm = thrust::transform_reduce(d_x.begin(), d_x.end(), unary_op, init, binary_op);

    std::cout << norm << std::endl;

    return 0;
    */
}