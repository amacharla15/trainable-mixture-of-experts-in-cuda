#include <cuda_runtime.h>
__global__ void matmul_naive_kernel(const float* A,const float* B,float* C,int M,int N,int K){
    float temp=0.0f;
    int row=blockIdx.y*blockDim.y+threadIdx.y;
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    if(row>=M || col>=N)return;
    for(int i=0;i<K;i++){
        temp+=A[row*K+i]*B[i*N+col];
    }
    C[row*N+col]=temp;
}
void launch_matmul_naive(const float* A,const float* B,float* C,int M,int N,int K){
    dim3 block(16,16);
    dim3 grid((N+15)/16,(M+15)/16);
    matmul_naive_kernel<<<grid,block>>>(A,B,C,M,N,K);
}