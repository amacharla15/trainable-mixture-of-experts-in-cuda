#include <cuda_runtime.h>
#include <math.h>
__global__ void gelu_backward_kernel(const float* x,const float* dy,float* dx,int n){
    int gid=blockIdx.x*blockDim.x+threadIdx.x;
    if(gid<n){
        float v=x[gid];
        float t=tanhf(0.79788456f*(v+0.044715f*v*v*v));
        dx[gid]=dy[gid]*(0.5f*(1.0f+t)+0.5f*v*(1.0f-t*t)*0.79788456f*(1.0f+0.134145f*v*v));
    }
}
void launch_gelu_backward(const float* x,const float* dy,float* dx,int n){
    if(n<=0)return;
    int threads=256;
    int blocks=(n+threads-1)/threads;
    gelu_backward_kernel<<<blocks,threads>>>(x,dy,dx,n);
}