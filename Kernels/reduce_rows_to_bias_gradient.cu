__global__ void reduce_rows_to_bias_grad_kernel(const float* dY, float* dbias, int M, int N) {
    int gid=blockIdx.x*blockDim.x+threadIdx.x;
    if(gid<N){
        float sum=0.0f;
        for(int i=0;i<M;i++){
            sum=sum+dY[i*N+gid];
        }
        dbias[gid]=sum;
    }
}
void launch_reduce_rows_to_bias_grad(const float* dY,float* dbias,int M,int N){
    if(M<=0 || N<=0)return;
    int threads=128;
    int blocks=(N+threads-1)/threads;
    reduce_rows_to_bias_grad_kernel<<<blocks,threads>>>(dY,dbias,M,N);
}