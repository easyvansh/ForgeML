#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#define CUDA_OK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ const char* n=cudaGetErrorName(e); const char* s=cudaGetErrorString(e); std::fprintf(stderr,"CUDA error %s:%d: code=%d name=%s description=%s\n",__FILE__,__LINE__,(int)e,n?n:"unknown",s?s:"unknown"); if(e==cudaErrorInsufficientDriver) std::fprintf(stderr,"CUDA runtime is newer than the installed NVIDIA driver; update the driver or compile with a matching older toolkit.\n"); return 2; } } while(0)

// One thread computes one (batch, head, query, channel) output. The score
// row is streamed with online max/sum updates, so no T x T score matrix is
// materialized. This is a correctness-first reference kernel, not FlashAttention.
__global__ void causal_attention(const float* q,const float* k,const float* v,float* out,int B,int H,int T,int D){
  int index=blockIdx.x*blockDim.x+threadIdx.x; int total=B*H*T*D; if(index>=total) return;
  int d=index%D, t=(index/D)%T, h=(index/(D*T))%H, b=index/(D*T*H); float scale=rsqrtf((float)D);
  float m=-INFINITY, l=0.0f, acc=0.0f; int qbase=(((b*H+h)*T+t)*D);
  for(int j=0;j<=t;j++){
    int base=(((b*H+h)*T+j)*D); float score=0.0f;
    for(int c=0;c<D;c++) score += q[qbase+c]*k[base+c]; score*=scale;
    float next=fmaxf(m,score); float old=(isinf(m))?0.0f:expf(m-next); float p=expf(score-next);
    acc=old*acc+p*v[base+d]; l=old*l+p; m=next;
  }
  out[index]=acc/l;
}

static float ref(const std::vector<float>&q,const std::vector<float>&k,const std::vector<float>&v,std::vector<float>&o,int B,int H,int T,int D){
  float maxerr=0; float scale=1.0f/std::sqrt((float)D);
  for(int b=0;b<B;b++)for(int h=0;h<H;h++)for(int t=0;t<T;t++){
    std::vector<float>s(t+1); float m=-INFINITY; for(int j=0;j<=t;j++){float z=0;int qb=(((b*H+h)*T+t)*D),kb=(((b*H+h)*T+j)*D);for(int d=0;d<D;d++)z+=q[qb+d]*k[kb+d];s[j]=z*scale;m=std::max(m,s[j]);}
    float den=0;for(float&z:s){z=std::exp(z-m);den+=z;} for(int d=0;d<D;d++){float z=0;for(int j=0;j<=t;j++){int vb=(((b*H+h)*T+j)*D);z+=s[j]*v[vb+d];}int ob=(((b*H+h)*T+t)*D)+d;o[ob]=z/den;}
  }
  return maxerr;
}

int main(int argc,char**argv){
  int T=argc>1?std::atoi(argv[1]):128,B=1,H=2,D=32; size_t n=(size_t)B*H*T*D; std::vector<float>q(n),k(n),v(n),cpu(n),gpu(n); for(size_t i=0;i<n;i++){q[i]=std::sin(i*.017f);k[i]=std::cos(i*.013f);v[i]=std::sin(i*.009f);}
  ref(q,k,v,cpu,B,H,T,D); float *dq,*dk,*dv,*do_; CUDA_OK(cudaMalloc(&dq,n*sizeof(float))); CUDA_OK(cudaMalloc(&dk,n*sizeof(float))); CUDA_OK(cudaMalloc(&dv,n*sizeof(float))); CUDA_OK(cudaMalloc(&do_,n*sizeof(float))); CUDA_OK(cudaMemcpy(dq,q.data(),n*sizeof(float),cudaMemcpyHostToDevice)); CUDA_OK(cudaMemcpy(dk,k.data(),n*sizeof(float),cudaMemcpyHostToDevice)); CUDA_OK(cudaMemcpy(dv,v.data(),n*sizeof(float),cudaMemcpyHostToDevice));
  int threads=128, blocks=(int)((n+threads-1)/threads); causal_attention<<<blocks,threads>>>(dq,dk,dv,do_,B,H,T,D); CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize()); CUDA_OK(cudaMemcpy(gpu.data(),do_,n*sizeof(float),cudaMemcpyDeviceToHost)); float err=0;for(size_t i=0;i<n;i++)err=std::max(err,std::abs(cpu[i]-gpu[i]));
  cudaEvent_t a,z;CUDA_OK(cudaEventCreate(&a));CUDA_OK(cudaEventCreate(&z));for(int i=0;i<10;i++){causal_attention<<<blocks,threads>>>(dq,dk,dv,do_,B,H,T,D);}CUDA_OK(cudaDeviceSynchronize());CUDA_OK(cudaEventRecord(a));for(int i=0;i<50;i++)causal_attention<<<blocks,threads>>>(dq,dk,dv,do_,B,H,T,D);CUDA_OK(cudaEventRecord(z));CUDA_OK(cudaEventSynchronize(z));float ms=0;CUDA_OK(cudaEventElapsedTime(&ms,a,z));
  std::printf("{\"context\":%d,\"batch\":%d,\"heads\":%d,\"head_dim\":%d,\"max_abs_error\":%.9g,\"median_like_ms\":%.6f,\"launches\":50,\"kernel\":\"causal_online_softmax_reference\"}\n",T,B,H,D,err,ms/50.0f); CUDA_OK(cudaFree(dq));CUDA_OK(cudaFree(dk));CUDA_OK(cudaFree(dv));CUDA_OK(cudaFree(do_)); return err<2e-5?0:1;
}
