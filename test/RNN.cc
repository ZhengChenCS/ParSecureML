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

float *labels;
float *images;
string imagesPath;
string labelsPath;
int isRand = 1;
int train_size = 640000;
int batch_size = 128;
int SIZE = 2048;
int row;
int col;
int FSIZE;

int dim1 = 128;
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
    if((images = (float*)malloc(sizeof(float)*train_size*SIZE)) == NULL){
        cout << "Malloc error." << endl;
        return 0;
    }
    if((labels = (float*)malloc(sizeof(float)*train_size)) == NULL){
        cout << "Malloc error." << endl;
        return 0;
    }
    if(isRand == 1){
        Generator(images, 256,  train_size*SIZE);
        Generator(labels, 10, train_size);
    }
    offline_stime = timestamp();
    if(MPI_rank == MPI_client){
        // read_Label(str_labels, labels);
        // read_Images(str_images, images);
        float *model1 = (float*)malloc(sizeof(float)*dim1*batch_size);
        for(int i = 0; i < dim1*batch_size; i++){
            model1[i] = (float)rand()/RAND_MAX;
        }
        float *model2 = (float*)malloc(sizeof(float)*dim1*dim1);
        for(int i = 0; i < dim1*dim1; i++){
            model2[i] = (float)rand()/RAND_MAX;
        }
        float *images_th1 = (float*)malloc(sizeof(float)*train_size*SIZE);
        float *images_th2 = (float*)malloc(sizeof(float)*train_size*SIZE);
        float *model1_th1 = (float*)malloc(sizeof(float)*dim1*batch_size);
        float *model1_th2 = (float*)malloc(sizeof(float)*dim1*batch_size);
        float *model2_th1 = (float*)malloc(sizeof(float)*dim1*dim1);
        float *model2_th2 = (float*)malloc(sizeof(float)*dim1*dim1);
        float *y_hat1 = (float*)malloc(sizeof(float)*train_size);
        float *y_hat2 = (float*)malloc(sizeof(float)*train_size);
        double c_mstime, c_metime;
        c_mstime = timestamp();
        for(int i = 0; i < train_size; i++){
            for(int j = 0; j < SIZE; j++){
                images_th1[i*SIZE+j] = rand() % 256;
                images_th2[i*SIZE+j] = images[i*SIZE+j] - images_th1[i*SIZE+j];	
            }
        }
        for(int i = 0; i < dim1*batch_size; i++){
            model1_th1[i] = (float)rand() /RAND_MAX;
            model1_th2[i] = model1[i] - model1_th1[i];
        }
        for(int i = 0; i < dim1*dim1; i++){
            model2_th1[i] = (float)rand() /RAND_MAX;
            model2_th2[i] = model2[i] - model2_th1[i];
        }
        for(int i = 0; i < train_size; i++){
            y_hat1[i] = rand()%10;
            y_hat2[i] = labels[i] - y_hat1[i];
        }	
        
        int flag1 = 0;
        int flag2 = -1;
        cout << "start..." << endl;
        cout << "sp start ..." << endl;
        Support sp1, sp2, sp3;
        sp1.GetShape(dim1, batch_size, batch_size, SIZE);
        sp1.Initial();
        sp1.Assign();
        //cout << "here" << endl;
        sp2.GetShape(dim1, batch_size, batch_size, SIZE);
        sp2.Initial();
        sp2.Assign();

        sp3.GetShape(dim1, dim1, dim1, SIZE);
        sp3.Initial();
        sp3.Assign();

        c_metime = timestamp();
        cout << "Client make time:" << c_metime - c_mstime << endl;
        MPI_Send(&flag1, 1, MPI_INT, MPI_server1, 0, MPI_COMM_WORLD);
        MPI_Send(images_th1, train_size*SIZE, MPI_FLOAT, MPI_server1, 0, MPI_COMM_WORLD);
        MPI_Send(model1_th1, dim1*batch_size, MPI_FLOAT, MPI_server1, 0, MPI_COMM_WORLD);
        MPI_Send(model2_th1, dim1*dim1, MPI_FLOAT, MPI_server1, 0, MPI_COMM_WORLD);
        MPI_Send(y_hat1, train_size, MPI_FLOAT, MPI_server1, 0, MPI_COMM_WORLD);
        
        MPI_Send(&flag2, 1, MPI_INT, MPI_server2, 0, MPI_COMM_WORLD);
        MPI_Send(images_th2, train_size*SIZE, MPI_FLOAT, MPI_server2, 0, MPI_COMM_WORLD);
        MPI_Send(model1_th2, dim1*batch_size, MPI_FLOAT, MPI_server2, 0, MPI_COMM_WORLD);
        MPI_Send(model2_th2, dim1*dim1, MPI_FLOAT, MPI_server2, 0, MPI_COMM_WORLD);
        MPI_Send(y_hat2, train_size, MPI_FLOAT, MPI_server2, 0, MPI_COMM_WORLD);
        //cout << "here" << endl;
        sp1.Send(MPI_server1, MPI_server2);
        sp2.Send(MPI_server1, MPI_server2);
        sp3.Send(MPI_server1, MPI_server2);
        cout << "Client:Send finished." << endl;
        free(images_th1);
        free(images_th2);
        free(model1_th1);
        free(model1_th2);
        free(model2_th1);
        free(model2_th2);
        free(y_hat1);
        free(y_hat2);
        sp1.Release();
        sp2.Release();
        sp3.Release();
    }
    else{
        double offline_etime;
        MPI_Status status;
        int flag;
        float y_hat[train_size];
        float *img = (float*)malloc(sizeof(float)*train_size*SIZE);
        float *model1 = (float*)malloc(sizeof(float)*dim1*batch_size);
        float *model2 = (float*)malloc(sizeof(float)*dim1*dim1);
        cout << "server" << MPI_rank << "..." << endl;
        MPI_Recv(&flag, 1, MPI_INT, MPI_client, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(img, train_size*SIZE, MPI_FLOAT, MPI_client, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(model1, dim1*batch_size, MPI_FLOAT, MPI_client, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(model2, dim1*dim1, MPI_FLOAT, MPI_client, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(y_hat, train_size, MPI_FLOAT, MPI_client, 0, MPI_COMM_WORLD, &status);
        Triplet T1, T2, T3;
        
        T1.GetShape(dim1, batch_size, batch_size, SIZE);
        T1.Initial();
        T2.GetShape(dim1, batch_size, batch_size, SIZE);
        T2.Initial();
        T3.GetShape(dim1, dim1, dim1, SIZE);
        T3.Initial();
        T1.Recv(MPI_client);
        T2.Recv(MPI_client);
        T3.Recv(MPI_client);
        offline_etime = timestamp();
        cout << "Server" << MPI_rank << ": offline end time:" << offline_etime-offline_stime << "." << endl;
        double online_stime, online_etime;
        
        int MPI_dest;
        if(MPI_rank == MPI_server1){
            MPI_dest = MPI_server2;
        }
        else{
            MPI_dest = MPI_server1;
        }
        float batchY[batch_size];
        float *batchImg = (float*)malloc(sizeof(float)*batch_size*SIZE);
        cout << batch_size << endl;
        online_stime = timestamp();
        for(int i = 0; i+batch_size <= train_size; i+=batch_size){
            SelectBatch(img, y_hat, batchImg, batchY, i, batch_size, SIZE);
            T1.GetData(model1, batchImg);
            T1.Rec(MPI_dest);
            T1.OP(flag);
            T2.GetData(model1, batchImg);
            T2.Rec(MPI_dest);
            T2.OP(flag);
            T3.GetData(model2, T1.C);
            T3.Rec(MPI_dest); 
            T3.OP(flag);
            for(int i = 0; i < dim1*SIZE; i++){
                T3.C[i] += T2.C[i];
            }
            //break;
        }
        T1.Release();
        T2.Release();
        T3.Release();
        online_etime = timestamp();
        cout << "Server" << MPI_rank << " train time:" << online_etime-online_stime << endl;
    }
    MPI_Finalize();
    all_etime = timestamp();
    cout << "All time:" << all_etime - all_stime << endl;
    return 0;
}
