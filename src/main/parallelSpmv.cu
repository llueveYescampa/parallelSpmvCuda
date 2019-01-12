#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "real.h"
#include "dataDef.h"

#include "parallelSpmv.h"

#define FATAL(msg) \
    do {\
        fprintf(stderr, "[%s:%d] %s\n", __FILE__, __LINE__, msg);\
        exit(-1);\
    } while(0)

#define REP 1000

#ifdef DOUBLE
    texture<int2>  xTex;
    texture<int2>  valTex;
#else
    texture<float> xTex;
    texture<float> valTex;
#endif

__global__ void spmv0(real *__restrict__ y, 
                      //real *__restrict__ x, 
                      //real *__restrict__ val,  
                      int  *__restrict__ row_ptr, 
                      int  *__restrict__ col_idx, 
                      const int nRows
                      );
                     
__global__ void spmv1(real *__restrict__ y, 
                      //real *__restrict__ x, 
                      //real *__restrict__ val,  
                      int  *__restrict__ row_ptr, 
                      int  *__restrict__ col_idx, 
                      const int nRows
                     );


real calculateSD(real *data, int n)
{
    real sum = (real) 0.0; 
    real mean;
    real standardDeviation = (real) 0.0;

    for(int i=0; i<n; ++i) {
        sum += data[i];
    } // end for //

    mean = sum/n;

    for(int i=0; i<n; ++i) {
        standardDeviation += pow(data[i] - mean, 2);
    } // end for //

    return sqrt(standardDeviation/n);
} // end of calculateSD //


int main(int argc, char *argv[]) 
{
 
    //int provided;
    //MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);

    const int root=0;
    int worldRank=0;
    //MPI_Comm_rank(MPI_COMM_WORLD,&worldRank);
    //MPI_Comm_size(MPI_COMM_WORLD,&worldSize);
    
    #include "parallelSpmvData.h"

    // verifing number of input parameters //
   char exists='t';
   char checkSol='f';
    if (worldRank == root) {
        if (argc < 3 ) {
            printf("Use: %s  Matrix_filename InputVector_filename  [SolutionVector_filename]  \n", argv[0]);     
            exists='f';
        } // endif //
        
        FILE *fh=NULL;
        // testing if matrix file exists
        if((fh = fopen(argv[1], "rb")  )   == NULL) {
            printf("No matrix file found.\n");
            exists='f';
        } // end if //
        
        // testing if input file exists
        if((fh = fopen(argv[2], "rb")  )   == NULL) {
            printf("No input vector file found.\n");
            exists='f';
        } // end if //

        // testing if output file exists
        if (argc  >3 ) {
            if((fh = fopen(argv[3], "rb")  )   == NULL) {
                printf("No output vector file found.\n");
                exists='f';
            } else {
                checkSol='t';
            } // end if //
        } // end if //
        if (fh) fclose(fh);
    } // end if //
    //MPI_Bcast(&exists,  1,MPI_CHAR,root,MPI_COMM_WORLD);
    if (exists == 'f') {
        if (worldRank == root) printf("Quitting.....\n");
        //MPI_Finalize();
        exit(0);
    } // end if //
    //MPI_Bcast(&checkSol,1,MPI_CHAR,root,MPI_COMM_WORLD);

    
    reader(&n_global,&nnz_global, &n, 
           &off_proc_nnz,
           &row_ptr,&col_idx,&val,
           &row_ptr_off,&col_idx_off,&val_off,
           argv[1], root);
    
    // ready to start //    

    real *w=NULL;
    real *v=NULL; // <-- input vector to be shared later
    //real *v_off=NULL; // <-- input vector to be shared later
    
    w     = (real *) malloc((n)*sizeof(real)); 
    v     = (real *) malloc((n)*sizeof(real));
    //v_off = (real *) malloc((nColsOff)*sizeof(real));

    // reading input vector
    vectorReader(v, &n, argv[2]);
//////////////////////////////////////
// cuda stuff start here

    /////////////////////////////////////////////////////
    // determining the standard deviation of the nnz per row
    real *temp=(real *) malloc((n_global)*sizeof(real));
    for (int row=0; row<n_global; ++row) {
        temp[row] = row_ptr[row+1] - row_ptr[row];
    } // end for //
    real sd=calculateSD(temp,n_global);
    free(temp);
    /////////////////////////////////////////////////////

    int *rows_d, *cols_d;
    real *vals_d;
    real *v_d, *w_d;
    cudaError_t cuda_ret;

    // Allocating device memory for input matrices 

    cuda_ret = cudaMalloc((void **) &rows_d,  (n_global+1)*sizeof(int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory for rows_d");
    
    cuda_ret = cudaMalloc((void **) &cols_d,  (nnz_global)*sizeof(int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory for cols_d");
    
    cuda_ret = cudaMalloc((void **) &vals_d,  (nnz_global)*sizeof(real));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory for vals_d");

    cuda_ret = cudaMalloc((void **) &v_d,  (n_global)*sizeof(real));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory for x_d");

    cuda_ret = cudaMalloc((void **) &w_d,  (n_global)*sizeof(real));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory for y_d");


    // Copy the input matrices from the host memory to the device memory

    cuda_ret = cudaMemcpy(rows_d, row_ptr, (n_global+1)*sizeof(int),cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device matrix rows_d");

    cuda_ret = cudaMemcpy(cols_d, col_idx, (nnz_global)*sizeof(int),cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device matrix cols_d");

    cuda_ret = cudaMemcpy(vals_d, val, (nnz_global)*sizeof(real),cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device matrix vals_d");

#ifdef USE_PIN_MEMORY
    cudaFreeHost(row_ptr);
    cudaFreeHost(col_idx);
    cudaFreeHost(val);
#else
    free(row_ptr);
    free(col_idx);
    free(val);
#endif
    cuda_ret = cudaMemcpy(v_d, v, (n_global)*sizeof(real),cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device matrix x_d");

    real meanNnzPerRow = ((real) nnz_global) / (n_global);
    
    const int basicSize = 32;
    dim3 block(basicSize);
    dim3 grid;
    const real parameter2Adjust = 0.5;

    if (meanNnzPerRow + parameter2Adjust*sd < basicSize) {
    	// these mean use use spmv0
        grid.x = ( (n_global + block.x -1) /block.x );
        printf("using spmv0: %f, %f, blockSize: %d\n", meanNnzPerRow, sd,block.x) ;        
    } else {
    	// these mean use use spmv1
        grid.x = n_global;
        if (meanNnzPerRow >= 2*basicSize) block.x = 2*basicSize;
        printf("using spmv1: %f, %f, blockSize: %d\n", meanNnzPerRow, sd,block.x) ;
    } // end if // 

    
    //printf("%d, %d\n", grid.x, block.x); exit(0);

    // Timing should begin here//
    struct timeval tp;                                   // timer
    double elapsed_time;
    
    gettimeofday(&tp,NULL);  // Unix timer
    elapsed_time = -(tp.tv_sec*1.0e6 + tp.tv_usec);
    
    for (int t=0; t<REP; ++t) {
        cuda_ret = cudaMemset(w_d, 0, (size_t) n_global*sizeof(real));
        if(cuda_ret != cudaSuccess) FATAL("Unable to copy set device matrix y_d");

        cuda_ret = cudaBindTexture(NULL, xTex, v_d, n_global*sizeof(real));
        cuda_ret = cudaBindTexture(NULL, valTex, vals_d, nnz_global*sizeof(real));
        if (meanNnzPerRow + parameter2Adjust*sd < basicSize) {
            spmv0<<<grid, block>>>(w_d, rows_d, cols_d, n_global);
        } else {
            spmv1<<<grid, block, block.x*sizeof(real)>>>(w_d,  rows_d, cols_d, n_global);
        } // end if // 
        cuda_ret = cudaUnbindTexture(xTex);
        cuda_ret = cudaUnbindTexture(valTex);

        cuda_ret = cudaMemcpy(w, w_d, (n_global)*sizeof(real),cudaMemcpyDeviceToHost);
        if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device matrix y_d back to host");
    } // end for //
    
    gettimeofday(&tp,NULL);
    elapsed_time += (tp.tv_sec*1.0e6 + tp.tv_usec);
    printf ("Total time was %f seconds.\n", elapsed_time*1.0e-6);

// cuda stuff ends here
//////////////////////////////////////
   
    if (checkSol=='t') {
        real *sol=NULL;
        sol     = (real *) malloc((n)*sizeof(real)); 
        // reading input vector
        vectorReader(sol, &n, argv[3]);
        
        int row=0;
        const real tolerance=1.0e-08;
        real error;
        do {
            error =  fabs(sol[row] - w[row]) /fabs(sol[row]);
            if ( error > tolerance ) break;
            ++row;
        } while (row < n); // end do-while //
        
        if (row == n) {
            printf("Solution match in rank %d\n",worldRank);
        } else {    
            printf("For Matrix %s, solution does not match at element %d in rank %d   %20.13e   -->  %20.13e  error -> %20.13e, tolerance: %20.13e \n", 
            argv[1], (row+1),worldRank, sol[row], w[row], error , tolerance  );
        } // end if //
        free(sol);    
    } // end if //

    free(w);
    free(v);
    
    #include "parallelSpmvCleanData.h" 
    //MPI_Finalize();
    return 0;    
} // end main() //
