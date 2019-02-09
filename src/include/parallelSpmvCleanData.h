    free(row_ptr_off);
    free(col_idx_off);
    free(val_off);

    free(w);
    free(v);
    
    cudaFree(rows_d);
    cudaFree(cols_d);
    cudaFree(vals_d);
    cudaFree(v_d);
    cudaFree(w_d);
    
