#include <cuda_runtime.h>
__global__ void relu_backward_kernel(const float* x,const float* dy,float* dx,int n){
    int gid=blockIdx.x*blockDim.x+threadIdx.x;
    if(gid<n){
        if(x[gid]>0){
            dx[gid]=dy[gid];
        }
        else{
            dx[gid]=0.0f;
        }
    }
}
void launch_relu_backward(const float* x,const float* dy,float* dx,int n){
    if(n<=0)return;
    int threads=256;
    int blocks=(n+threads-1)/threads;
    relu_backward_kernel<<<blocks,threads>>>(x,dy,dx,n);
}