void reader(int *gn, int *gnnz, int *n,  int *off_proc_nnz, 
            int **rPtr,int **cIdx,real **v,int **rPtrO,int **cIdxO,real **vO,
            const char *matrixFile, const int root);

void vectorReader(real *v, const int *n, const char *vectorFile);
int createColIdxMap(int **b,  int *a, const int *n);

__global__ 
void spmv(real *__restrict__ y, 
          //real *__restrict__ x, 
          //real *__restrict__ val,  
          int  *__restrict__ row_ptr, 
          int  *__restrict__ col_idx, 
          const int nRows
          );

