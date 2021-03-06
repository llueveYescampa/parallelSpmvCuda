#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "real.h"

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

    printf("%s Precision.\n", (sizeof(real) == sizeof(double)) ? "Double": "Single");

    
    reader(&n_global,&nnz_global, &n, 
           &off_proc_nnz,
           &row_ptr,&col_idx,&val,
           &row_ptr_off,&col_idx_off,&val_off,
           argv[1], root);
    
    // ready to start //    
    cudaError_t cuda_ret;


    real *w=NULL;
    real *v=NULL; // <-- input vector to be shared later
    //real *v_off=NULL; // <-- input vector to be shared later
    
    
    v     = (real *) malloc(n*sizeof(real));
    w     = (real *) malloc(n*sizeof(real)); 
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


    cuda_ret = cudaMemcpy(v_d, v, (n_global)*sizeof(real),cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device matrix x_d");

    free(row_ptr);
    free(col_idx);
    free(val);

    real meanNnzPerRow = ((real) nnz_global) / (n_global);

    const int warpSize = 32;
    dim3 block, grid;
    const real parameter2Adjust = 0.15;
    size_t sharedMemorySize=0;

    if (meanNnzPerRow < 10 && parameter2Adjust*sd < warpSize) {
        if (meanNnzPerRow < (real) 4.5) {
            block.x=128;
        } else if (meanNnzPerRow < (real) 14.4) {
            block.x=64;
        } else {
            block.x=32;
        } // end if //
    	// these mean use scalar spmv
        grid.x = ( (n_global + block.x -1) /block.x );
        printf("using scalar spmv, blockSize: [%d, %d] %f, %f\n",block.x,block.y, meanNnzPerRow, sd) ;
    } else {
        // these mean use vector spmv
        if (meanNnzPerRow > 10.0*warpSize) {
            block.x=2*warpSize;
        }  else if (meanNnzPerRow > 4.0*warpSize) {
            block.x=warpSize/2;
        }  else {
            block.x=warpSize/4;
        } // end if //
        block.y=128/block.x;
        grid.x = ( (n_global + block.y - 1) / block.y ) ;
        printf("using vector spmv, blockSize: [%d, %d] %f, %f\n",block.x,block.y, meanNnzPerRow, sd) ;
    	sharedMemorySize=block.x*block.y*sizeof(real);
    } // end if // 
    
    //printf("%d, %d, %d \n", grid.x, block.x, block.y); exit(0);

    // Timing should begin here//
    struct timeval tp;                                   // timer
    double elapsed_time;
    
    gettimeofday(&tp,NULL);  // Unix timer
    elapsed_time = -(tp.tv_sec*1.0e6 + tp.tv_usec);
    
    for (int t=0; t<REP; ++t) {

        cuda_ret = cudaMemset(w_d, 0, (size_t) n_global*sizeof(real));
        if(cuda_ret != cudaSuccess) FATAL("Unable to set device for matrix w_d");

        cuda_ret = cudaBindTexture(NULL, xTex, v_d, n_global*sizeof(real));
        cuda_ret = cudaBindTexture(NULL, valTex, vals_d, nnz_global*sizeof(real));
        spmv<<<grid, block, sharedMemorySize>>>(w_d,  rows_d, cols_d, n_global);
        cuda_ret = cudaUnbindTexture(xTex);
        cuda_ret = cudaUnbindTexture(valTex);
        //cudaDeviceSynchronize();
        cudaStreamSynchronize(NULL);
        
    } // end for //
    
    gettimeofday(&tp,NULL);
    elapsed_time += (tp.tv_sec*1.0e6 + tp.tv_usec);
    printf ("Total time was %f seconds, GFLOPS: %f\n", elapsed_time*1.0e-6, 2.0*nnz_global*REP*1.0e-3/elapsed_time);

    cuda_ret = cudaMemcpy(w, w_d, (n_global)*sizeof(real),cudaMemcpyDeviceToHost);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device matrix y_d back to host");

// cuda stuff ends here
//////////////////////////////////////
   
    if (checkSol=='t') {
        real *sol=NULL;
        sol     = (real *) malloc((n)*sizeof(real)); 
        // reading input vector
        vectorReader(sol, &n, argv[3]);
        
        int row=0;
        real tolerance = 1.0e-08;
        if (sizeof(real) != sizeof(double) ) {
            tolerance = 1.0e-02;
        } // end if //
        
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

    
    #include "parallelSpmvCleanData.h" 
    //MPI_Finalize();
    return 0;    
} // end main() //
