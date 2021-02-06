#include <omp.h>
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
#include <random>
#include <thread>
#include "../include/ParSecureML.h"
#include "../include/read.h"
#include "mpi.h"
using namespace std;

#if defined (_MSC_VER)  // Visual studio
    #define thread_local __declspec( thread )
#elif defined (__GCC__) // GCC
    #define thread_local __thread
#endif

#define FLOAT_PER_CACHELINE (64/4)
const int N = 16384;
/*
double timestamp(){
    struct timeval time;
    gettimeofday(&time, NULL);
    return time.tv_sec + 1e-6*time.tv_usec;
}*/


/* Thread-safe function that returns a random number between min and max (inclusive).
This function takes ~142% the time that calling rand() would take. For this extra
cost you get a better uniform distribution and thread-safety. */
extern int intRand(const int & min, const int & max, bool lastUsed = false);

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


        int procNum = omp_get_num_procs();
        int *seed = (int *)malloc(sizeof(int) * procNum*2);
        printf("Proc Numb: %d\n", procNum);
        for (int i = 0; i < procNum*2; i++){
            seed[i] = rand();
        }
#pragma omp parallel num_threads(2*procNum)
{
        int s = seed[omp_get_thread_num()];
#pragma omp for
    	for(int i = 0; i < N*N; i++){
            A1[i] = intRand(0, 255);
            A2[i] = A[i] - A1[i];
        }
#pragma omp for
        for(int i = 0; i < N*N; i++){
            B1[i] = intRand(0, 255);
            B2[i] = B[i] - B1[i];
        } 
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
#pragma omp parallel for
        for(int i = 0; i < N*N/FLOAT_PER_CACHELINE; i++){
            for (int j = 0; j < FLOAT_PER_CACHELINE; j++){
                C[i * FLOAT_PER_CACHELINE + j] = C1[i * FLOAT_PER_CACHELINE + j] + C2[i * FLOAT_PER_CACHELINE + j];
            }
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
