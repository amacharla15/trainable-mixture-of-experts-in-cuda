#include <cuda_runtime.h>
__global__ void elementwise_add_kernel(const float* a,const float* b,float* out,int n){
    int gid=blockIdx.x*blockDim.x+threadIdx.x;
    if(gid<n){
        out[gid]=a[gid]+b[gid];
    }
}
void launch_elementwise_add(const float* a,const float* b,float* out,int n){
    if(n<=0)return;
    int threads=256;
    int blocks=(n+threads-1)/threads;
    elementwise_add_kernel<<<blocks,threads>>>(a,b,out,n);
}