#include <iostream>
#include <fstream>
#include <ostream>
#include <istream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <assert.h>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <cstring>
#include "../include/ParSecureML.h"
#include "../include/read.h"
#include "mpi.h"
using namespace std;

const int N = 4096;

double timestamp(){
    struct timeval time;
    gettimeofday(&time, NULL);
    return time.tv_sec + 1e-6*time.tv_usec;
}

int main(int argc, char **argv){
    MPI_Init(&argc, &argv);
    int MPI_rank, MPI_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &MPI_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &MPI_size);
    double all_stime, all_etime;
    all_stime = timestamp();
    double offline_stime;
    
    offline_stime = timestamp();
    if(MPI_rank == MPI_client){
        float *A, *B, *C;
        
        A = (float*)malloc(sizeof(float)*N*N);
        B = (float*)malloc(sizeof(float)*N*N);
        C = (float*)malloc(sizeof(float)*N*N);
        Generator(A, 256, N*N);
        Generator(B, 256, N*N);
        float *A1 = (float*)malloc(sizeof(float)*N*N);
        float *A2 = (float*)malloc(sizeof(float)*N*N);
        float *B1 = (float*)malloc(sizeof(float)*N*N);
        float *B2 = (float*)malloc(sizeof(float)*N*N);
        float *C1 = (float*)malloc(sizeof(float)*N*N);
        float *C2 = (float*)malloc(sizeof(float)*N*N);
        for(int i = 0; i < N*N; i++){
            A1[i] = rand()%256;
            A2[i] = A[i] - A1[i];
        }
        for(int i = 0; i < N*N; i++){
            B1[i] = rand()%256;
            B2[i] = B[i] - B1[i];
        } 
        cout << "Client..." << endl;
        Support sp;
        sp.GetShape(N, N, N, N);
        sp.Initial();
        sp.Assign();
        int flag1 = 0;
        int flag2 = -1;
        cout << "Client send..." << endl;
        MPI_Send(&flag1, 1, MPI_INT, MPI_server1, 0, MPI_COMM_WORLD);
        MPI_Send(A1, N*N, MPI_FLOAT, MPI_server1, 0, MPI_COMM_WORLD);
        MPI_Send(B1, N*N, MPI_FLOAT, MPI_server1, 0, MPI_COMM_WORLD);

        MPI_Send(&flag2, 1, MPI_INT, MPI_server2, 0, MPI_COMM_WORLD);
        MPI_Send(A2, N*N, MPI_FLOAT, MPI_server2, 0, MPI_COMM_WORLD);
        MPI_Send(B2, N*N, MPI_FLOAT, MPI_server2, 0, MPI_COMM_WORLD);

        sp.Send(MPI_server1, MPI_server2);
        cout << "Client: Send finished." << endl;
        MPI_Status status;
        MPI_Recv(C1, N*N, MPI_FLOAT, MPI_server1, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(C2, N*N, MPI_FLOAT, MPI_server2, 0, MPI_COMM_WORLD, &status);
        for(int i = 0; i < N*N; i++){
            C[i] = C1[i] + C2[i];
        }
        cout << "finished." << endl;
        free(A1);
        free(A2);
        free(B1);
        free(B2);
        free(A);
        free(B);
        free(C);
        sp.Release();
    }
    else{
        double offline_etime;
        MPI_Status status;
        int flag;
        float *A1 = (float*)malloc(sizeof(float)*N*N);
        float *B1 = (float*)malloc(sizeof(float)*N*N);
        cout << "server" << MPI_rank << "..." << endl;
        MPI_Recv(&flag, 1, MPI_INT, MPI_client, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(A1, N*N, MPI_FLOAT, MPI_client, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(B1, N*N, MPI_FLOAT, MPI_client, 0, MPI_COMM_WORLD, &status);
        Triplet T;
        T.GetShape(N, N, N, N);
        T.Initial();
        T.Recv(MPI_client);
        offline_etime = timestamp();
        cout << "Server" << MPI_rank << ": offline end time:" << offline_etime-offline_stime << "." << endl;

        double online_stime, online_etime;
        online_stime = timestamp();
        int MPI_dest;
        if(MPI_rank == MPI_server1){
            MPI_dest = MPI_server2;
        }
        else{
            MPI_dest = MPI_server1;
        }
        T.GetData(A1, B1);
        T.Rec(MPI_dest);
        T.OP(flag);
        online_etime = timestamp();
        cout << "Server" << MPI_rank << " online phase time:" << online_etime-online_stime << endl;
        cout << "Server" << MPI_rank << " send to client..." << endl;

        MPI_Send(T.C, N*N, MPI_FLOAT, MPI_client, 0, MPI_COMM_WORLD);
        T.Release();
    }
    MPI_Finalize();
    all_etime = timestamp();
    cout << "All time:" << all_etime - all_stime << endl;
    return 0;
}
