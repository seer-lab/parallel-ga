#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>

#define MAXTHREADS 2
#define NBBLOCKS 2


__global__ 
void testRand ( curandState *state) {
    int idx = threadIdx.x  + blockIdx.x * blockDim.x;



    float myrandf = curand_uniform(&state[idx]);
    myrandf *= ((5*blockIdx.x+5) - (5*blockIdx.x)-1+0.999999);
    myrandf += (5*blockIdx.x);
    int myrand = (int)truncf(myrandf);

    printf("Id %i, block id: %i value %d\n",idx,blockIdx.x,myrand);

}
__global__
void setup_kernel (curandState* state, unsigned long seed )
{
    int id = threadIdx.x  + blockIdx.x * blockDim.x;
    curand_init ( seed, id , 0, &state[id] );
}


int main() {
    const dim3 blockSize(MAXTHREADS);
    const dim3 gridSize(NBBLOCKS);

    curandState* devStates;
    cudaMalloc ( &devStates,MAXTHREADS*NBBLOCKS*sizeof( curandState ) );
    time_t t;
    time(&t);
    setup_kernel <<< gridSize, blockSize >>> ( devStates, (unsigned long) t );  
    testRand  <<< gridSize, blockSize >>> ( devStates);  
    testRand  <<< gridSize, blockSize >>> ( devStates);  

    cudaFree(devStates);
}