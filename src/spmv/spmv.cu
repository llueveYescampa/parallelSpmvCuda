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
void spmv(real *__restrict__ y, 
           //real *__restrict__ x, 
           //real *__restrict__ val, 
           int  *__restrict__ row_ptr, 
           int  *__restrict__ col_idx, 
           const int nRows
          )
{   
    if (blockDim.y==1) { 
        const int row = blockIdx.x*blockDim.x + threadIdx.x;
        if (row < nRows)  {
            real dot = (real) 0.0;
            for ( int col = row_ptr[row]; col < row_ptr[row+1]; ++col ) {
                //dot += (val[col] * x[col_idx[col]]);
                dot += (fetch_real(valTex,col) * fetch_real( xTex, col_idx[col])); 
            } // end for //
            y[row] = dot;
        } // end if //
    } else {    
        extern __shared__ real temp[];
        const unsigned int row = blockIdx.x*blockDim.y + threadIdx.y;
        const unsigned int sharedMemIndx = blockDim.x*threadIdx.y + threadIdx.x;
        temp[sharedMemIndx] = (real) 0.0;

        if (row < nRows) {
            for (unsigned int col=row_ptr[row]+threadIdx.x; col < row_ptr[row+1]; col+=blockDim.x) {
                //temp[threadIdx.x] += (val[col] * x[col_idx[col]]);
                temp[ sharedMemIndx] += (fetch_real(valTex,col) * fetch_real( xTex, col_idx[col]));
            } // end for //
            __syncthreads();

            if (blockDim.x == 64) {
                if (threadIdx.x<32) temp[sharedMemIndx] += temp[sharedMemIndx + 32];
                __syncthreads();
            } // end if //
          
            // unrolling warp 
            if (threadIdx.x < 16) {
                volatile real *temp1 = temp;
                temp1[sharedMemIndx] += temp1[sharedMemIndx + 16];
                temp1[sharedMemIndx] += temp1[sharedMemIndx + 8];
                temp1[sharedMemIndx] += temp1[sharedMemIndx + 4];
                temp1[sharedMemIndx] += temp1[sharedMemIndx + 2];
                temp1[sharedMemIndx] += temp1[sharedMemIndx + 1];
            } // end if //

            if ((sharedMemIndx % blockDim.x)  == 0) {
                y[row] = temp[sharedMemIndx];
            } // end if //   
        } // end if
    } // end if //
    
} // end of spmv() //
