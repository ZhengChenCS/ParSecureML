/*
This file contains some functions for compression communiate
*/

#include "../include/compress.h"
#include <cstdlib>
#include <string.h>
void MallocCsr(struct CSR &C, int val_count, int row, int col){
    C.val = (float*)malloc(sizeof(float)*val_count);
	C.col = (int*)malloc(sizeof(float)*val_count);
	C.row = (int*)malloc(sizeof(float)*row);
}
void ReleaseCsr(struct CSR &C){
    free(C.val);
    free(C.col);
    free(C.row);
}
void Compress(float *matrix, struct CSR &C, int val_count, int row, int col){
    int count = 0;
	for(int i = 0; i < row; i++){
		int flag = 1;
		for(int j = 0; j < col; j++){
			if(abs(matrix[i*col+j]) < 1e-3){
				continue;
			}else{
				C.val[count] = matrix[i*col+j];
				C.col[count++] = j;
				if(flag == 1){
					C.row[i] = j;
					flag = 0;
				}
			}
		}
		if(flag==1){
			C.row[i] = -1;
		}
	}
}
void deCompress(struct CSR &C, int val_count, int row, int col, float *mat){
	int cur = 0;
	for(int i = 0; i < row; i++){
		if(C.row[i] == -1) continue;
		for(int j = C.row[i]; j < C.row[i+1]; j++){
			mat[i*col+j] += C.val[cur++];
		}
	}
}