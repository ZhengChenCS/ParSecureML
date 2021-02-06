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
int train_size;
int batch_size;
int SIZE;
int row;
int col;
int FSIZE;

int dim1 = 512*512;
int dim2 = 508*508;
int dim3 = 64;
int dim4 = 10;
int main(int argc, char **argv){
    MPI_Init(&argc, &argv);
    int MPI_rank, MPI_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &MPI_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &MPI_size);
    double all_stime, all_etime;
    all_stime = timestamp();
    double offline_stime;
    string setupFile = argv[1];
    cout << "setup file: " << setupFile << endl;
    ifstream setup(setupFile);
    if(!setup.is_open()){
        cout << "Setup not exist." << endl;
        return 0;
    }
    string line;
    while(getline(setup, line)){
        stringstream para(line);
        string attr;
        para >> attr;
        if(attr == "--imagesPath"){
            para >> imagesPath;
            isRand = 0;
        }
        else if(attr == "--labelsPath"){
            para >> labelsPath;
        }
        else if(attr == "--train_size"){
            para >> train_size;
        }
        else if(attr == "--batch_size"){
            para >> batch_size;
        }
        else if(attr == "--SIZE"){
            para >> SIZE;
        }
        else if(attr == "--row"){
            para >> row;
        }
        else if(attr == "--col"){
            para >> col;
        }
        else if(attr == "--FSIZE"){
            para >> FSIZE;
        }
        else{
            cout << "Setup parameters error." << endl;
            return 0;
        }
    }
    setup.close();
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
    }else{
        read_Images(imagesPath, images, SIZE, train_size);
        Generator(labels, 10, train_size);
    }
    offline_stime = timestamp();
    if(MPI_rank == MPI_client){
        // read_Label(str_labels, labels);
        // read_Images(str_images, images);
        float *model1 = (float*)malloc(sizeof(float)*FSIZE*FSIZE);
        float *model2 = (float*)malloc(sizeof(float)*dim2*dim3);
        float *model3 = (float*)malloc(sizeof(float)*dim3*dim4);
        for(int i = 0; i < FSIZE*FSIZE; i++){
            model1[i] = (float)rand()/RAND_MAX;
        }
        for(int i = 0; i < dim2*dim3; i++){
            model2[i] = (float)rand()/RAND_MAX;
        }
        for(int i = 0; i < dim3*dim4; i++){
            model3[i] = (float)rand()/RAND_MAX;
        }
        float *images_th1 = (float*)malloc(sizeof(float)*train_size*SIZE);
        float *images_th2 = (float*)malloc(sizeof(float)*train_size*SIZE);
        float *model1_th1 = (float*)malloc(sizeof(float)*FSIZE*FSIZE);
        float *model1_th2 = (float*)malloc(sizeof(float)*FSIZE*FSIZE);
        float *model2_th1 = (float*)malloc(sizeof(float)*dim2*dim3);
        float *model2_th2 = (float*)malloc(sizeof(float)*dim2*dim3);
        float *model3_th1 = (float*)malloc(sizeof(float)*dim3*dim4);
        float *model3_th2 = (float*)malloc(sizeof(float)*dim3*dim4);
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
        for(int i = 0; i < FSIZE*FSIZE; i++){
            model1_th1[i] = (float)rand() /RAND_MAX;
            model1_th2[i] = model1[i] - model1_th1[i];
        }
        for(int i = 0; i < dim2*dim3; i++){
            model2_th1[i] = (float)rand() /RAND_MAX;
            model2_th2[i] = model2[i] - model2_th1[i];
        }
        for(int i = 0; i < dim3*dim4; i++){
            model3_th1[i] = (float)rand() /RAND_MAX;
            model3_th2[i] = model3[i] - model3_th1[i];
        }
        for(int i = 0; i < train_size; i++){
            y_hat1[i] = rand()%10;
            y_hat2[i] = labels[i] - y_hat1[i];
        }	
        
        int flag1 = 0;
        int flag2 = -1;
        cout << "start..." << endl;
        cout << "sp start ..." << endl;
        ConvSupport sp1;
        Support sp2, sp3;
        sp1.GetShape(FSIZE, FSIZE);
        sp1.Initial();
        sp1.Assign();
        
        sp2.GetShape(batch_size, dim2, dim2, dim3);
        sp2.Initial();
        sp2.Assign();

        sp3.GetShape(batch_size, dim3, dim3, dim4);
        sp3.Initial();
        sp3.Assign();

        // sp4.GetShape(dim4, 1, 1, batch_size);
        // sp4.Initial();
        // sp4.Assign();

        // sp5.GetShape(dim4, batch_size, batch_size, dim3);
        // sp5.Initial();
        // sp5.Assign();

        // sp6.GetShape(dim3, dim4, dim4, batch_size);
        // sp6.Initial();
        // sp6.Assign();

        // sp7.GetShape(dim3, batch_size, batch_size, dim2);
        // sp7.Initial();
        // sp7.Assign();

        // sp8.GetShape(dim2, dim3, dim3, batch_size);
        // sp8.Initial();
        // sp8.Assign();

        // sp9.GetShape(dim2, batch_size, batch_size, dim1);
        // sp9.Initial();
        // sp9.Assign();
        c_metime = timestamp();
        cout << "Client make time:" << c_metime - c_mstime << endl;
        MPI_Send(&flag1, 1, MPI_INT, MPI_server1, 0, MPI_COMM_WORLD);
        MPI_Send(images_th1, train_size*SIZE, MPI_FLOAT, MPI_server1, 0, MPI_COMM_WORLD);
        MPI_Send(model1_th1, FSIZE*FSIZE, MPI_FLOAT, MPI_server1, 0, MPI_COMM_WORLD);
        MPI_Send(model2_th1, dim2*dim3, MPI_FLOAT, MPI_server1, 0, MPI_COMM_WORLD);
        MPI_Send(model3_th1, dim3*dim4, MPI_FLOAT, MPI_server1, 0, MPI_COMM_WORLD);
        MPI_Send(y_hat1, train_size, MPI_FLOAT, MPI_server1, 0, MPI_COMM_WORLD);
        
        MPI_Send(&flag2, 1, MPI_INT, MPI_server2, 0, MPI_COMM_WORLD);
        MPI_Send(images_th2, train_size*SIZE, MPI_FLOAT, MPI_server2, 0, MPI_COMM_WORLD);
        MPI_Send(model1_th2, FSIZE*FSIZE, MPI_FLOAT, MPI_server2, 0, MPI_COMM_WORLD);
        MPI_Send(model2_th2, dim2*dim3, MPI_FLOAT, MPI_server2, 0, MPI_COMM_WORLD);
        MPI_Send(model3_th2, dim3*dim4, MPI_FLOAT, MPI_server2, 0, MPI_COMM_WORLD);
        MPI_Send(y_hat2, train_size, MPI_FLOAT, MPI_server2, 0, MPI_COMM_WORLD);
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
        free(model3_th1);
        free(model3_th2);
        free(y_hat1);
        free(y_hat2);
        sp1.Release();
        sp2.Release();
        sp3.Release();
        // sp4.Release();
        // sp5.Release();
        // sp6.Release();
        // sp7.Release();
        // sp8.Release();
        // sp9.Release();
    }
    else{
        double offline_etime;
        MPI_Status status;
        int flag;
        float y_hat[train_size];
        float *img = (float*)malloc(sizeof(float)*train_size*SIZE);
        float *model1 = (float*)malloc(sizeof(float)*FSIZE*FSIZE);
        float *model2 = (float*)malloc(sizeof(float)*dim2*dim3);
        float *model3 = (float*)malloc(sizeof(float)*dim3*dim4);
        cout << "server" << MPI_rank << "..." << endl;
        MPI_Recv(&flag, 1, MPI_INT, MPI_client, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(img, train_size*SIZE, MPI_FLOAT, MPI_client, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(model1, FSIZE*FSIZE, MPI_FLOAT, MPI_client, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(model2, dim2*dim3, MPI_FLOAT, MPI_client, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(model3, dim3*dim4, MPI_FLOAT, MPI_client, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(y_hat, train_size, MPI_FLOAT, MPI_client, 0, MPI_COMM_WORLD, &status);
        ConvTriplet layer1;
        Triplet layer2, layer3;
        layer1.GetShape(row, col, FSIZE, FSIZE, batch_size);
        layer1.Initial();
        layer2.GetShape(batch_size, dim2, dim2, dim3);
        layer2.Initial();
        layer3.GetShape(batch_size, dim3, dim3, dim4);
        layer3.Initial();
        layer1.Recv(MPI_client);
        cout << "here" << endl;
        layer2.Recv(MPI_client);
        layer3.Recv(MPI_client);
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
        float batchY[batch_size];
        float *batchImg = (float*)malloc(sizeof(float)*batch_size*SIZE);
        cout << batch_size << endl;
        for(int i = 0; i+batch_size <= train_size; i+=batch_size){
            //cout << "BatchCur:" << i << endl;
            SelectBatch(img, y_hat, batchImg, batchY, i, batch_size, SIZE);
            layer1.GetData(batchImg, model1);
            layer1.Rec(MPI_dest);
            layer1.OP(flag);
            layer2.GetData(layer1.C, model2);
            
            layer2.Rec(MPI_dest);
            layer2.OP(flag);
            layer3.GetData(layer2.C, model3);
            layer3.Rec(MPI_dest);
            layer3.OP(flag);
            //break;
        }
        online_etime = timestamp();
cout << "Comm time: " << layer1.commtime << endl;

        layer1.Release();
        layer2.Release();
        layer3.Release();
        free(img);
        free(model1);
        free(model2);
        free(model3);
        free(batchImg);
        
        cout << "Server" << MPI_rank << " train time:" << online_etime-online_stime << endl;
    }
    MPI_Finalize();
    all_etime = timestamp();
    cout << "All time:" << all_etime - all_stime << endl;
    return 0;
}
