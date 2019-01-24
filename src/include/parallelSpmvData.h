    int n_global,nnz_global;
    int n;
    int off_proc_nnz=0;

    // data for the on_proc solution
    int *row_ptr=NULL;
    int *col_idx=NULL;
    real *val=NULL;
    // end of data for the on_proc solution
    
    // data for the off_proc solution
    int *row_ptr_off=NULL;
    int *col_idx_off=NULL;
    real *val_off=NULL;
    // end of data for the off_proc solution
    
    cudaStream_t stream;

