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

