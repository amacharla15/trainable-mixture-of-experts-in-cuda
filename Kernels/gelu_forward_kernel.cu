#include <cuda_runtime.h>
#include <math.h>
__global__ void gelu_forward_kernel(const float* x,float* y,int n){
    int gid=blockIdx.x*blockDim.x+threadIdx.x;
    if(gid<n){
        float v=x[gid];
        y[gid]=0.5f*v*(1.0f+tanhf(0.79788456f*(v+0.044715f*v*v*v)));
    }
}
void launch_gelu_forward(const float* x,float* y,int n){
    if(n<=0)return;
    int threads=256;
    int blocks=(n+threads-1)/threads;
    gelu_forward_kernel<<<blocks,threads>>>(x,y,n);
}