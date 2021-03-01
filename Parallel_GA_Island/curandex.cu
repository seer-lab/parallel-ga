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
void testRand ( curandState *state, int nb ) {
    int idx = threadIdx.x  + blockIdx.x * blockDim.x;

    printf("Id %i, value %f\n",idx,-5.12 + (curand_uniform(&state[idx])) * (5.12 - -5.12));

}
__global__
void setup_kernel (curandState* state, unsigned long seed )
{
    int id = threadIdx.x  + blockIdx.x * blockDim.x;
    curand_init ( seed, id , 0, &state[id] );
}

/**
* Image comes in in horizontal lines
*/
int main() {
    const dim3 blockSize(MAXTHREADS);
    const dim3 gridSize(NBBLOCKS);

    curandState* devStates;
    cudaMalloc ( &devStates,MAXTHREADS*NBBLOCKS*sizeof( curandState ) );
    time_t t;
    time(&t);
    setup_kernel <<< gridSize, blockSize >>> ( devStates, (unsigned long) t );  
    int nb = 4;
    testRand  <<< gridSize, blockSize >>> ( devStates,nb);  
    testRand  <<< gridSize, blockSize >>> ( devStates,nb);  

    cudaFree(devStates);
}