#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "gputimer.h"

__global__
void reduce_kernal(float *d_in, float *d_out){
    int indx = blockDim.x * blockIdx.x + threadIdx.x;
    int tindx = threadIdx.x;

    for(int i = blockDim.x / 2; i > 0; i >>= 1){
        if(tindx < i){
            d_in[indx] = fmin(d_in[indx], d_in[indx + i]);
        }
        __syncthreads();
    }

    if(tindx == 0){
        d_out[blockIdx.x] = d_in[indx];
    }
}

__global__
void reduce_kernal_shared_mem(float *d_in, float *d_out){
    int indx = blockDim.x * blockIdx.x + threadIdx.x;
    int tindx = threadIdx.x;

    extern __shared__ float sh_in[];

    sh_in[tindx] = -99999.0f;

    sh_in[tindx] = d_in[indx];
    __syncthreads();

    for(int i = blockDim.x / 2; i > 0; i >>= 1){
        if(tindx < i){
            sh_in[tindx] = fmax(sh_in[tindx], sh_in[tindx + i]);
        }
        __syncthreads();
    }

    if(tindx == 0){
        d_out[blockIdx.x] = sh_in[0];
    }
}

void reduce(float *d_in, float *d_int, float *d_out, const int ARRAY_SIZE, bool is_shared){
    if(!is_shared){
        reduce_kernal<<<1024, 1024>>>(d_in, d_int);
        reduce_kernal<<<1, 1024>>>(d_int, d_out);
    }else{
        reduce_kernal_shared_mem<<<1024, 1024, 1024 * sizeof(float)>>>(d_in, d_int);
        reduce_kernal_shared_mem<<<1, 1024, 1024 * sizeof(float)>>>(d_int, d_out);
    }
}   

int main(int argc, char ** argv){

    const int ARRAY_SIZE = 1 << 20;
    const int ARRAY_SIZE_BYTES = ARRAY_SIZE * sizeof(float);
    
    float h_in[ARRAY_SIZE];

    float sum = 0.0f;
    float maxx = 0.0f;
    //fill array
    for(int i = 0; i < ARRAY_SIZE; i++){
        h_in[i] =  i * 1.0f;
        sum += h_in[i];
        maxx = max(maxx, h_in[i]);
    }

    float *d_in, *d_out, *d_int;
    
    cudaMalloc(&d_in, ARRAY_SIZE_BYTES);
    cudaMalloc(&d_out, ARRAY_SIZE_BYTES);
    cudaMalloc(&d_int, ARRAY_SIZE_BYTES);

    cudaMemcpy(d_in, h_in, ARRAY_SIZE_BYTES, cudaMemcpyHostToDevice);
    
    GpuTimer gpuTimer;

    gpuTimer.Start();
    reduce(d_in, d_int, d_out, ARRAY_SIZE, true);
    gpuTimer.Stop();
    printf("Elapsed Time : %f\n", gpuTimer.Elapsed());
    cudaDeviceSynchronize();

    float h_out = 0.0f;

    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("  GPU : %f \n  CPU : %f\n", h_out, maxx);

    cudaFree(&d_in);
    cudaFree(&d_out);
    cudaFree(&d_int);
    
    return 0;
}