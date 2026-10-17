__global__ void matmul_tiled_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float a[16][16];
    __shared__ float b[16][16];
    int row=blockIdx.y*blockDim.y+threadIdx.y;
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    int row_tid=threadIdx.y;
    int col_tid=threadIdx.x;
    float temp=0.0f;
    for(int tile=0;tile<K;tile+=16){
        if(row<M && tile+col_tid<K){
            a[row_tid][col_tid]=A[row*K+tile+col_tid];
        }else{
            a[row_tid][col_tid]=0.0f;
        }
        if(tile+row_tid<K && col<N){
            b[row_tid][col_tid]=B[(tile+row_tid)*N+col];
        }else{
            b[row_tid][col_tid]=0.0f;
        }
        __syncthreads();
        for(int i=0;i<16;i++){
            temp=temp+a[row_tid][i]*b[i][col_tid];
        }
        __syncthreads();
    }
    if(row<M && col<N){
        C[row*N+col]=temp;
    }
}