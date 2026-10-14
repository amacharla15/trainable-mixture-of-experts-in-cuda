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

