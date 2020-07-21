#include <iostream>
#include <cstdlib>
#include <cstring>
#include "../include/ParSecureML.h"
#include "mpi.h"
void Support::GetShape(int in_row1, int in_col1, int in_row2, int in_col2){
	row1 = in_row1;
	col1 = in_col1;
	row2 = in_row2;
	col2 = in_col2;
}
void Support::Initial(){
	U1 = (float*)malloc(sizeof(float)*row1*col1);
	U2 = (float*)malloc(sizeof(float)*row1*col1);
	U = (float*)malloc(sizeof(float)*row1*col1);
	V1 = (float*)malloc(sizeof(float)*row2*col2);
	V2 = (float*)malloc(sizeof(float)*row2*col2);
	V = (float*)malloc(sizeof(float)*row2*col2);
	Z1 = (float*)malloc(sizeof(float)*row1*col2);
	Z2 = (float*)malloc(sizeof(float)*row1*col2);
	Z = (float*)malloc(sizeof(float)*row1*col2);
	MallocD(GPU_U, row1*col1);
	MallocD(GPU_V, row2*col2);
	MallocD(GPU_Z, row1*col2);
}
void Support::Assign(){
	for(int i = 0; i < row1*col1; i++){
		U1[i] = (float)rand()/RAND_MAX;
		U2[i] = (float)rand()/RAND_MAX;
		U[i] = U1[i] + U2[i];
	}
	for(int i = 0; i < row2*col2; i++){
		V1[i] = (float)rand()/RAND_MAX;
		V2[i] = (float)rand()/RAND_MAX;
		V[i] = V1[i] + V2[i];
	}
	CopyHtoD(GPU_U, U, row1*col1);
	CopyHtoD(GPU_V, V, row2*col2);
	GPU_Mul();
	CopyDtoH(Z, GPU_Z, row1*col2);
    for(int i = 0; i < row1*col2; i++){
		Z1[i] = (float)rand()/RAND_MAX;
		Z2[i] = Z[i] - Z1[i];
	}
}
void Support::Send(int MPI_dest1, int MPI_dest2){
	MPI_Send(U1, row1*col1, MPI_FLOAT, MPI_dest1, 0, MPI_COMM_WORLD);
	MPI_Send(V1, row2*col2, MPI_FLOAT, MPI_dest1, 0, MPI_COMM_WORLD);
	MPI_Send(Z1, row1*col2, MPI_FLOAT, MPI_dest1, 0, MPI_COMM_WORLD);

	MPI_Send(U2, row1*col1, MPI_FLOAT, MPI_dest2, 0, MPI_COMM_WORLD);
	MPI_Send(V2, row2*col2, MPI_FLOAT, MPI_dest2, 0, MPI_COMM_WORLD);
	MPI_Send(Z2, row1*col2, MPI_FLOAT, MPI_dest2, 0, MPI_COMM_WORLD);
}

void Support::Release(){
	free(U1);
	free(U2);
	free(U);
	free(V1);
	free(V2);
	free(V);
	free(Z);
    free(Z1);
    free(Z2);
    ReleaseGPU(GPU_U);
    ReleaseGPU(GPU_V);
    ReleaseGPU(GPU_Z);
}
void ConvSupport::GetShape(int in_row, int in_col){
	row = in_row;
	col = in_col;
}
void ConvSupport::Initial(){
	U1 = (float*)malloc(sizeof(float)*row*col);
	U2 = (float*)malloc(sizeof(float)*row*col);
	U = (float*)malloc(sizeof(float)*row*col);
	V1 = (float*)malloc(sizeof(float)*row*col);
	V2 = (float*)malloc(sizeof(float)*row*col);
	V = (float*)malloc(sizeof(float)*row*col);
	Z1 = (float*)malloc(sizeof(float)*row*col);
	Z2 = (float*)malloc(sizeof(float)*row*col);
	Z = (float*)malloc(sizeof(float)*row*col);
}
void ConvSupport::Assign(){
	for(int i = 0; i < row*col; i++){
		U1[i] = (float)rand()/RAND_MAX;
		U2[i] = (float)rand()/RAND_MAX;
		U[i] = U1[i] + U2[i];
	}
	for(int i = 0; i < row*col; i++){
		V1[i] = (float)rand()/RAND_MAX;
		V2[i] = (float)rand()/RAND_MAX;
		V[i] = V1[i] + V2[i];
	}
	for(int i = 0; i < row; i++){
		for(int j = 0; j < col; j++){
			Z[i*col+j] = U[i*col+j] * V[i*col+j];
		}
	}
    for(int i = 0; i < row*col; i++){
		Z1[i] = (float)rand()/RAND_MAX;
		Z2[i] = Z[i] - Z1[i];
	}
}
void ConvSupport::Send(int MPI_dest1, int MPI_dest2){
	MPI_Send(U1, row*col, MPI_FLOAT, MPI_dest1, 0, MPI_COMM_WORLD);
	MPI_Send(V1, row*col, MPI_FLOAT, MPI_dest1, 0, MPI_COMM_WORLD);
	MPI_Send(Z1, row*col, MPI_FLOAT, MPI_dest1, 0, MPI_COMM_WORLD);

	MPI_Send(U2, row*col, MPI_FLOAT, MPI_dest2, 0, MPI_COMM_WORLD);
	MPI_Send(V2, row*col, MPI_FLOAT, MPI_dest2, 0, MPI_COMM_WORLD);
	MPI_Send(Z2, row*col, MPI_FLOAT, MPI_dest2, 0, MPI_COMM_WORLD);
}

void ConvSupport::Release(){
	free(U1);
	free(U2);
	free(U);
	free(V1);
	free(V2);
	free(V);
	free(Z);
    free(Z1);
    free(Z2);
}
