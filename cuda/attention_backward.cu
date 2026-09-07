#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_OK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){std::fprintf(stderr,"CUDA error %s:%d code=%d name=%s description=%s\n",__FILE__,__LINE__,(int)e,cudaGetErrorName(e),cudaGetErrorString(e)); return 2;} } while(0)

__global__ void attention_backward(const float* q,const float* k,const float* v,
    const float* dout,float* dq,float* dk,float* dv,int B,int H,int T,int D) {
  int idx=blockIdx.x*blockDim.x+threadIdx.x, total=B*H*T*D; if(idx>=total)return;
  int d=idx%D, row=idx/D, t=row%T, bh=row/T; int qb=(bh*T+t)*D;
  float scale=rsqrtf((float)D), maxs=-INFINITY;
  for(int j=0;j<=t;j++){float s=0;for(int c=0;c<D;c++)s+=q[qb+c]*k[(bh*T+j)*D+c];maxs=fmaxf(maxs,s*scale);}
  float z=0; for(int j=0;j<=t;j++){float s=0;for(int c=0;c<D;c++)s+=q[qb+c]*k[(bh*T+j)*D+c];z+=expf(s*scale-maxs);}
  float dot=0; for(int j=0;j<=t;j++){float s=0;for(int c=0;c<D;c++)s+=q[qb+c]*k[(bh*T+j)*D+c];float p=expf(s*scale-maxs)/z;float dp=0;for(int c=0;c<D;c++)dp+=dout[qb+c]*v[(bh*T+j)*D+c];dot+=p*dp;}
  float gq=0;
  for(int j=0;j<=t;j++){
    int base=(bh*T+j)*D; float s=0;for(int c=0;c<D;c++)s+=q[qb+c]*k[base+c];float p=expf(s*scale-maxs)/z;float dp=0;for(int c=0;c<D;c++)dp+=dout[qb+c]*v[base+c];float ds=p*(dp-dot);
    gq+=ds*k[base+d]*scale; atomicAdd(&dk[base+d],ds*q[qb+d]*scale); atomicAdd(&dv[base+d],p*dout[qb+d]);
  }
  dq[qb+d]=gq;
}

static void reference(const std::vector<float>&q,const std::vector<float>&k,const std::vector<float>&v,const std::vector<float>&dout,std::vector<float>&dq,std::vector<float>&dk,std::vector<float>&dv,int B,int H,int T,int D){
  float scale=1/std::sqrt((float)D);
  for(int bh=0;bh<B*H;bh++)for(int t=0;t<T;t++){
    int qb=(bh*T+t)*D; std::vector<float> p(t+1),dp(t+1); float mx=-INFINITY,z=0;
    for(int j=0;j<=t;j++){float s=0;for(int c=0;c<D;c++)s+=q[qb+c]*k[(bh*T+j)*D+c];mx=std::max(mx,s*scale);}
    for(int j=0;j<=t;j++){float s=0;for(int c=0;c<D;c++)s+=q[qb+c]*k[(bh*T+j)*D+c];p[j]=std::exp(s*scale-mx);z+=p[j];}
    for(float& x:p)x/=z; float dot=0;
    for(int j=0;j<=t;j++){for(int c=0;c<D;c++)dp[j]+=dout[qb+c]*v[(bh*T+j)*D+c];dot+=p[j]*dp[j];}
    for(int j=0;j<=t;j++){int base=(bh*T+j)*D;float ds=p[j]*(dp[j]-dot);for(int c=0;c<D;c++){dq[qb+c]+=ds*k[base+c]*scale;dk[base+c]+=ds*q[qb+c]*scale;dv[base+c]+=p[j]*dout[qb+c];}}
  }
}

int main(int argc,char**argv){int T=argc>1?std::atoi(argv[1]):127,B=1,H=2,D=32;size_t n=(size_t)B*H*T*D;std::vector<float>q(n),k(n),v(n),dout(n),cq(n),ck(n),cv(n),gq(n),gk(n),gv(n);for(size_t i=0;i<n;i++){q[i]=std::sin(i*.017f);k[i]=std::cos(i*.013f);v[i]=std::sin(i*.009f);dout[i]=std::cos(i*.011f);}reference(q,k,v,dout,cq,ck,cv,B,H,T,D);float*dq,*dk,*dv,*d_q,*d_k,*d_v;CUDA_OK(cudaMalloc(&d_q,n*4));CUDA_OK(cudaMalloc(&d_k,n*4));CUDA_OK(cudaMalloc(&d_v,n*4));CUDA_OK(cudaMalloc(&dq,n*4));CUDA_OK(cudaMalloc(&dk,n*4));CUDA_OK(cudaMalloc(&dv,n*4));CUDA_OK(cudaMemcpy(dq,q.data(),n*4,cudaMemcpyHostToDevice));CUDA_OK(cudaMemcpy(dk,k.data(),n*4,cudaMemcpyHostToDevice));CUDA_OK(cudaMemcpy(dv,v.data(),n*4,cudaMemcpyHostToDevice));CUDA_OK(cudaMemset(d_q,0,n*4));CUDA_OK(cudaMemset(d_k,0,n*4));CUDA_OK(cudaMemset(d_v,0,n*4));float*dd;CUDA_OK(cudaMalloc(&dd,n*4));CUDA_OK(cudaMemcpy(dd,dout.data(),n*4,cudaMemcpyHostToDevice));int threads=256,blocks=(int)((n+threads-1)/threads);attention_backward<<<blocks,threads>>>(dq,dk,dv,dd,d_q,d_k,d_v,B,H,T,D);CUDA_OK(cudaGetLastError());CUDA_OK(cudaDeviceSynchronize());CUDA_OK(cudaMemcpy(gq.data(),d_q,n*4,cudaMemcpyDeviceToHost));CUDA_OK(cudaMemcpy(gk.data(),d_k,n*4,cudaMemcpyDeviceToHost));CUDA_OK(cudaMemcpy(gv.data(),d_v,n*4,cudaMemcpyDeviceToHost));float e=0;for(size_t i=0;i<n;i++){e=std::max(e,std::fabs(gq[i]-cq[i]));e=std::max(e,std::fabs(gk[i]-ck[i]));e=std::max(e,std::fabs(gv[i]-cv[i]));}std::printf("{\"context\":%d,\"max_abs_error\":%.8g,\"kernel\":\"causal_attention_backward_reference\"}\n",T,e);return e<3e-5f?0:1;}
