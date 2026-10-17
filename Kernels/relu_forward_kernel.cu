#include <cuda_runtime.h>
__global__ void relu_forward_kernel(const float* x,float* y,int n){
    int gid=blockIdx.x*blockDim.x+threadIdx.x;
    if(gid<n){
        if(x[gid]<0){
            y[gid]=0.0f;
        }
        else{
            y[gid]=x[gid];
        }
    }
}
void launch_relu_forward(const float* x,float* y,int n){
    if(n<=0)return;
    int threads=256;
    int blocks=(n+threads-1)/threads;
    relu_forward_kernel<<<blocks,threads>>>(x,y,n);
}