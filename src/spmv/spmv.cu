#include <stdio.h>
#include "real.h"


#ifdef DOUBLE
    extern texture<int2> xTex;
    extern texture<int2> valTex;
#else
    extern texture<float> xTex;
    extern texture<float> valTex;
#endif

#ifdef DOUBLE
    static __inline__ __device__ 
    double fetch_real(texture<int2> t, int i)
    {
	    int2 v = tex1Dfetch(t,i);
	    return __hiloint2double(v.y, v.x);
    } // end of fetch_real() //
#else
    static __inline__ __device__ 
    float fetch_real(texture<float> t, int i)
    {
	    return tex1Dfetch(t,i);
    } // end of fetch_double() //
#endif

__global__ 
void spmv0(real *__restrict__ y, 
           //real *__restrict__ x, 
           //real *__restrict__ val, 
           int  *__restrict__ row_ptr, 
           int  *__restrict__ col_idx, 
           const int nRows
          )
{    
    const unsigned int row = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (row < nRows)  {
        real dot = (real) 0;
        for (int col = row_ptr[row]; col < row_ptr[row+1]; ++col ) {
            //dot += (val[col] * x[col_idx[col]]);
            dot += (fetch_real(valTex,col) * fetch_real( xTex, col_idx[col])); 
        } // end for //
        y[row] = dot;
    } // end if //
} // end of spmv0() //

__global__ 
void spmv1(real *__restrict__ y, 
           //real *__restrict__ x, 
           //real *__restrict__ val, 
           int *__restrict__  row_ptr, 
           int *__restrict__  col_idx, 
           int nRows)
{    
    extern __shared__ real temp[];
    temp[threadIdx.x] = (real) 0;

    const int row = blockIdx.x;
    
    for (int col=row_ptr[row]+threadIdx.x; col < row_ptr[row+1]; col+=blockDim.x) {
        //temp[threadIdx.x] += (val[col] * x[col_idx[col]]);
        temp[threadIdx.x] += (fetch_real(valTex,col) * fetch_real( xTex, col_idx[col]));
    } // end for //
    __syncthreads();
    
    // local reduction per block
    for (int next = blockDim.x/2; next > 0; next >>= 1 ) {
        if (threadIdx.x < next) {
            temp[threadIdx.x] += temp[threadIdx.x+next];
        } // end if // 
        __syncthreads();
    } // end for //

    if (threadIdx.x == 0) {
        y[blockIdx.x] = temp[0];
    } // end if //   
} // end of spmv1() //
