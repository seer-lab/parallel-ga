#include "evaluation.cuh"

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

void evaluation(int warp, double* h_population, double* h_fitness, double *d_population, double *d_fitness, const int row, const int col, size_t bytesPopulation, size_t bytesFitness) {

    // Copying memory onto device
    cudaMemcpy(d_population, h_population, bytesPopulation, cudaMemcpyHostToDevice);

    // Threads per block
    int TPB_SIZE = row/warp;

    sphere_eval<<<warp, TPB_SIZE>>>(d_population, d_fitness, row, col);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    // Copy back to host
    cudaMemcpy(h_fitness, d_fitness, bytesFitness, cudaMemcpyDeviceToHost);

}