#include <stdio.h>
#include <stdlib.h> /* for rand() */
#include <iostream>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

using namespace std;

bool errorAsk(const char *s="n/a")
{
cudaError_t err=cudaGetLastError();
if(err==cudaSuccess)
    return false;
printf("CUDA error [%s]: %s\n",s,cudaGetErrorString(err));
return true;
};

double *fillArray(double *c_idata,int N,double constant) {
    int n;
    for (n = 0; n < N; n++) {
            c_idata[n] = constant*floor(drand48()*10.0);

    }
return c_idata;
}

int main(int argc,char *argv[])
{
    int N;
    N = 100;

    double *c_data,*g_data;
//    result = new double[N];

    c_data = new double[N];
    c_data = fillArray(c_data,N,1.0);
    c_data[32] = -1.0;
    cudaMalloc(&g_data,N*sizeof(double));
    cudaMemcpy(g_data,c_data,N*sizeof(double),cudaMemcpyHostToDevice);
   // thrust::device_ptr<double> g_ptr =  thrust::device_pointer_cast(g_data);

    int result_offset = thrust::min_element(thrust::device, g_data, g_data + N);

    double min_value = *result_offset;
    // we could also do this:
    // double min_value = c_data[result_offset];
    std::cout<< "min value found at position: " << result_offset << " value: " << min_value << std::endl;
}