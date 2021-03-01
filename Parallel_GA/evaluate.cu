#include <vector>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include "timer.h"


__device__
double sphere(double x) { return x * x; }

__global__
void gpu_eval(double *p, double *f, int numRows, int numCols) {

    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    for (unsigned int i = 0; i < numCols; i++) {
        f[tid] += sphere(p[tid * numCols + i]);
    }

}

// Evaluates the score of each individual within a population
void evaluation (std::vector<std::vector<double>> population, std::vector<double> &fitness, const int p, const int populationSize) {

    GpuTimer timer;
    timer.Start();

    size_t bytes_fitness = populationSize * sizeof(double);
    size_t bytes_population = populationSize * p * sizeof(double);

    double *h_population, *h_fitness;
    double *d_population, *d_fitness;

    h_population = (double*)malloc(bytes_population);
    h_fitness = (double*)malloc(bytes_fitness);
    cudaMalloc(&d_population, bytes_population);
    cudaMalloc(&d_fitness, bytes_fitness);

    for (std::vector<int>::size_type i = 0; i < populationSize; i++)
        for (std::vector<int>::size_type j = 0; j < p; j++)
            h_population[i * p + j] = population[i][j];

    cudaMemcpy(d_population, h_population, bytes_population, cudaMemcpyHostToDevice);

    int TB_SIZE = populationSize/128;

    gpu_eval<<<128, TB_SIZE>>>(d_population, d_fitness, populationSize, p);

    cudaMemcpy(h_fitness, d_fitness, bytes_fitness, cudaMemcpyDeviceToHost);

    cudaFree(d_population);
    cudaFree(d_fitness);

    timer.Stop();

    for (std::vector<int>::size_type i = 0; i < populationSize; i++)
        fitness[i] = h_fitness[i];

    int err = printf("Your code ran in: %f msecs.\n", timer.Elapsed());

    if (err < 0) {
        //Couldn't print! Probably the student closed stdout - bad news
        std::cerr << "Couldn't print timing information! STDOUT Closed!" << std::endl;
        exit(1);
    }

} 