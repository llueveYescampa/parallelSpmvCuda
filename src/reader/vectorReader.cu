#include <stdio.h>
//#include <stdlib.h>

#include "real.h"
void vectorReader( real *v, const int *n, const char *vectorFile)
{
    //int worldRank, worldSize;
    //MPI_Comm_rank(MPI_COMM_WORLD,&worldRank);
    //MPI_Comm_size(MPI_COMM_WORLD,&worldSize);
    //MPI_Status status;
    
    int acumulate=0;
    const size_t offset = (acumulate )* sizeof(real) ;

    // opening vector file to read values
    FILE *filePtr;
    filePtr = fopen(vectorFile, "rb");
    // reading cols vector (n) values //
    fseek(filePtr, offset, SEEK_SET);
    
    if (sizeof(real) == sizeof(double)) {
        if ( !fread(v, sizeof(real), (size_t) *n, filePtr)) exit(0);
    } else {
        double *temp = (double *) malloc(*n*sizeof(double)); 
        if ( !fread(temp, sizeof(double), (size_t) (*n), filePtr)) exit(0);
        for (int i=0; i<*n; i++) {
            v[i] = (real) temp[i];
        } // end for //    
        free(temp);
    } // end if //
    
    fclose(filePtr);
    // end of opening vector file to read values
} // end of vectoReader //
