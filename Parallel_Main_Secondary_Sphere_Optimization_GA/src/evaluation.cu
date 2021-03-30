#include "../include/evaluation.cuh"

# define M_PI           3.14159265358979323846  /* pi */

__device__
double square(double x) { return x * x; }

__global__
void sphere_eval(double *p, double *f, int numRows, int numCols) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    // Return if we're out of bounds
    if (tid >= numRows)
        return;

    f[tid] = 0;
    __syncthreads();
    
    for (unsigned int i = 0; i < numCols; i++)
        f[tid] += square(p[tid * numCols + i]);

}

__global__
void rastrigin_eval(double *p, double *f, int numRows, int numCols) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    // Return if we're out of bounds
    if (tid >= numRows)
        return;

    f[tid] = 0;
    __syncthreads();
    
    for (unsigned int i = 0; i < numCols; i++)
        f[tid] += ((p[tid * numCols + i]*p[tid * numCols + i]) - (10*cos(2*M_PI*p[tid * numCols + i])) +10);

}

__global__
void ackley_eval(double *p, double *f, int numRows, int numCols) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    double a = 20.0, b = 0.2, c = 2*M_PI;
    double s1 = 0.0, s2 = 0.0;

    // Return if we're out of bounds
    if (tid >= numRows)
        return;

    f[tid] = 0;
    __syncthreads();
    

    for (unsigned int i = 0; i < numCols; i++) {
        s1 = s1 + pow(p[tid * numCols + i], 2);
        s2 = s2 + cos(c*p[tid * numCols + i]);   
    }
    
    f[tid] = -a * exp ( -b * rsqrt (s1 / double (numCols))) - exp (s2 / double (numCols)) + a + exp(1.0);

}

__global__
void griewank_eval(double *p, double *f, int numRows, int numCols) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    double s = 0.0, t = 0.0;

    // Return if we're out of bounds
    if (tid >= numRows)
        return;

    f[tid] = 0;
    __syncthreads();
    

    for (unsigned int i = 0; i < numCols; i++) {
        s += pow(p[tid * numCols + i], 2);
        t *= cos(p[tid * numCols + i] / rsqrt(double (i+1)));   
    }
    
    f[tid] = (s / 4000.0) - t + 1;

}


void evaluation(const int warp, double* h_population, double* h_fitness, double *d_population, double *d_fitness, 
                const int populationSize, const int p, size_t bytesPopulation, size_t bytesFitness) {

    // Copying memory onto device
    cudaMemcpy(d_population, h_population, bytesPopulation, cudaMemcpyHostToDevice);

    // Threads per block
    const int TPB_SIZE = populationSize/warp;

    ackley_eval<<<warp, TPB_SIZE>>>(d_population, d_fitness, populationSize, p);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    // Copy back to host
    cudaMemcpy(h_fitness, d_fitness, bytesFitness, cudaMemcpyDeviceToHost);

}