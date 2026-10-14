"""
Trainable Mixture of Experts in CUDA

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - matmul_naive_kernel
__global__ void matmul_naive_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    float temp = 0.0f;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    for (int i = 0; i < K; i++) {
        temp += A[row * K + i] * B[i * N + col];
    }

    C[row * N + col] = temp;
}

# Step 5 - add_bias_row_kernel
__global__ void add_bias_row_kernel(float* Y,const float* bias,int M,int N){
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    int row=blockIdx.y*blockDim.y+threadIdx.y;
    if(col<N && row<M){
        Y[row*N+col]+=bias[col];
    }
}

# Step 7 - elementwise_add_kernel
__global__ void elementwise_add_kernel(const float* a, const float* b, float* out, int n) {
    // TODO: compute out[i] = a[i] + b[i] for i in [0, n)
    int gid = blockIdx.x*blockDim.x+threadIdx.x;
    if(gid<n){
        out[gid]=a[gid]+b[gid];
    }
}

# Step 8 - relu_forward_kernel
__global__ void relu_forward_kernel(const float* x, float* y, int n) {
    // TODO: write y[i] = max(0, x[i]) for i in [0, n)
    int gid = blockIdx.x*blockDim.x+threadIdx.x;
    if(gid<n){
        if(x[gid]<0){
            y[gid]=0.0f;
        }
        else{
            y[gid]=x[gid];
        }
    }

}

# Step 9 - relu_backward_kernel
__global__ void relu_backward_kernel(const float* x, const float* dy, float* dx, int n) {
    // TODO: implement ReLU backward: dx[i] = dy[i] if x[i] > 0 else 0
    int gid = blockIdx.x*blockDim.x+threadIdx.x;
    if(gid<n){
        if(x[gid]>0){
            dx[gid]=dy[gid];
        }
        else if (x[gid]<0){
            dx[gid]=0.0f;
        }
    }
}

# Step 10 - gelu_forward_kernel
__global__ void gelu_forward_kernel(const float* x, float* y, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < n) {
        float val = x[gid];
        y[gid] = 0.5f * val * (1.0f + tanhf(0.79788456f * (val + 0.044715f * val * val * val)));
    }
}

# Step 11 - gelu_backward_kernel
__global__ void gelu_backward_kernel(const float* x,const float* dy,float* dx,int n){
    int gid=blockIdx.x*blockDim.x+threadIdx.x;
    if(gid<n){
        float v=x[gid];
        float t=tanhf(0.79788456f*(v+0.044715f*v*v*v));
        dx[gid]=dy[gid]*(0.5f*(1.0f+t)+0.5f*v*(1.0f-t*t)*0.79788456f*(1.0f+0.134145f*v*v));
    }
}

