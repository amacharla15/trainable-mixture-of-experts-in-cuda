__global__ void matmul_at_b_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row=blockIdx.y*blockDim.y+threadIdx.y;
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    if(row<M && col<N){
        float temp=0.0f;
        for(int i=0;i<K;i++){
            temp+=A[i*M+row]*B[i*N+col];
        }
        C[row*N+col]=temp;
    }
}
void launch_matmul_at_b(const float* A,const float* B,float* C,int M,int N,int K){
    if(M<=0 || N<=0 || K<=0)return;
    dim3 threads(16,16);
    dim3 blocks((N+15)/16,(M+15)/16);
    matmul_at_b_kernel<<<blocks,threads>>>(A,B,C,M,N,K);
}