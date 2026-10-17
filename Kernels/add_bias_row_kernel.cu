#include <cuda_runtime.h>
__global__ void add_bias_row_kernel(float* Y,const float* bias,int M,int N){
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    int row=blockIdx.y*blockDim.y+threadIdx.y;
    if(col<N && row<M){
        Y[row*N+col]+=bias[col];
    }
}
void launch_add_bias_row(float* Y,const float* bias,int M,int N){
    if(M<=0 || N<=0)return;
    dim3 block(16,16);
    dim3 grid((N+15)/16,(M+15)/16);
    add_bias_row_kernel<<<grid,block>>>(Y,bias,M,N);
}