#include <arpa/inet.h>
#include <assert.h>
#include <netinet/in.h>
#include <pthread.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <istream>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>
#include "../include/ParSecureML.h"
#include "../include/read.h"
#include "mpi.h"
using namespace std;

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define OUT_SIZE 50

float* labels;
float* images;
string imagesPath;
string labelsPath;
int isRand = 1;
int train_size;
int batch_size;
int SIZE;
int row;
int col;
int FSIZE;
float alpha;

int dim1 = 784;
int dim2 = 784;
int dim3 = 64;
int dim4 = 10;
/*
double timestamp(){
    struct timeval time;
    gettimeofday(&time, NULL);
    return time.tv_sec + 1e-6*time.tv_usec;
}
*/
float clipAlpha(float aj, float H, float L) {
    if (aj > H)
        return H;
    if (aj < L)
        return L;
    return aj;
}

int main(int argc, char** argv) {
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
    if (!setup.is_open()) {
        cout << "Setup not exist." << endl;
        return 0;
    }
    string line;
    while (getline(setup, line)) {
        stringstream para(line);
        string attr;
        para >> attr;
        if (attr == "--imagesPath") {
            para >> imagesPath;
            isRand = 0;
        } else if (attr == "--labelsPath") {
            para >> labelsPath;
        } else if (attr == "--train_size") {
            para >> train_size;
        } else if (attr == "--batch_size") {
            para >> batch_size;
        } else if (attr == "--SIZE") {
            para >> SIZE;
        } else if (attr == "--row") {
            para >> row;
        } else if (attr == "--col") {
            para >> col;
        } else if (attr == "--FSIZE") {
            para >> FSIZE;
        } else if (attr == "--alpha") {
            para >> alpha;
        } else {
            cout << "Setup parameters error." << endl;
            return 0;
        }
    }
    setup.close();
    batch_size = train_size;
    if ((images = (float*)malloc(sizeof(float) * train_size * SIZE)) == NULL) {
        cout << "Malloc error." << endl;
        return 0;
    }
    if ((labels = (float*)malloc(sizeof(float) * train_size)) == NULL) {
        cout << "Malloc error." << endl;
        return 0;
    }
    if (isRand == 1) {
        Generator(images, 256, train_size * SIZE);
        Generator(labels, 10, train_size);
    } else {
        read_Images(imagesPath, images, SIZE, train_size);
        Generator(labels, 10, train_size);
    }
    // cout << train_size << endl;
    // cout << SIZE << endl;
    offline_stime = timestamp();
    if (MPI_rank == MPI_client) {
        // read_Label(str_labels, labels);
        // read_Images(str_images, images);
        float* alphas = (float*)malloc(sizeof(float) * train_size * OUT_SIZE);
        for (int i = 0; i < train_size * OUT_SIZE; i++) {
            alphas[i] = 0;
        }
        float* images_th1 = (float*)malloc(sizeof(float) * train_size * SIZE);
        float* images_th2 = (float*)malloc(sizeof(float) * train_size * SIZE);
        float* alphas1 = (float*)malloc(sizeof(float) * train_size * OUT_SIZE);
        float* alphas2 = (float*)malloc(sizeof(float) * train_size * OUT_SIZE);
        float* y_hat1 = (float*)malloc(sizeof(float) * train_size);
        float* y_hat2 = (float*)malloc(sizeof(float) * train_size);
        double c_mstime, c_metime;

        for (int i = 0; i < train_size; i++) {
            for (int j = 0; j < SIZE; j++) {
                images_th1[i * SIZE + j] = rand() % 256;
                images_th2[i * SIZE + j] = images[i * SIZE + j] - images_th1[i * SIZE + j];
            }
        }

        for (int i = 0; i < train_size * OUT_SIZE; i++) {
            alphas1[i] = 0;
            alphas2[i] = 0;
        }
        for (int i = 0; i < train_size; i++) {
            y_hat1[i] = rand() % 10;
            y_hat2[i] = labels[i] - y_hat1[i];
        }

        int flag1 = 0;
        int flag2 = -1;
        cout << "start..." << endl;
        cout << "sp start ..." << endl;
        c_mstime = timestamp();
        Support sp1, sp2, sp3, sp4, sp5, sp6, sp7, sp8, sp9;

        sp1.GetShape(batch_size, SIZE, SIZE, 1);
        sp1.Initial();
        sp1.Assign();
        sp2.GetShape(OUT_SIZE, batch_size, batch_size, 1);
        sp2.Initial();
        sp2.Assign();
        sp3.GetShape(OUT_SIZE, SIZE, SIZE, OUT_SIZE);
        sp3.Initial();
        sp3.Assign();

        c_metime = timestamp();
        cout << "Client make time:" << c_metime - c_mstime << endl;
        MPI_Send(&flag1, 1, MPI_INT, MPI_server1, 0, MPI_COMM_WORLD);
        MPI_Send(images_th1, train_size * SIZE, MPI_FLOAT, MPI_server1, 0, MPI_COMM_WORLD);
        MPI_Send(alphas1, train_size * OUT_SIZE, MPI_FLOAT, MPI_server1, 0, MPI_COMM_WORLD);
        MPI_Send(y_hat1, train_size, MPI_FLOAT, MPI_server1, 0, MPI_COMM_WORLD);

        MPI_Send(&flag2, 1, MPI_INT, MPI_server2, 0, MPI_COMM_WORLD);
        MPI_Send(images_th2, train_size * SIZE, MPI_FLOAT, MPI_server2, 0, MPI_COMM_WORLD);
        MPI_Send(alphas2, train_size * OUT_SIZE, MPI_FLOAT, MPI_server2, 0, MPI_COMM_WORLD);
        MPI_Send(y_hat2, train_size, MPI_FLOAT, MPI_server2, 0, MPI_COMM_WORLD);
        sp1.Send(MPI_server1, MPI_server2);
        sp2.Send(MPI_server1, MPI_server2);
        sp3.Send(MPI_server1, MPI_server2);

        cout << "Client:Send finished." << endl;
        MPI_Status status;
        MPI_Recv(alphas1, train_size * OUT_SIZE, MPI_FLOAT, MPI_server1, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(alphas2, train_size * OUT_SIZE, MPI_FLOAT, MPI_server2, 0, MPI_COMM_WORLD, &status);
        for (int i = 0; i < train_size * OUT_SIZE; i++) {
            alphas[i] = alphas1[i] + alphas2[i];
        }
        cout << "finished." << endl;
        free(images_th1);
        free(images_th2);
        free(alphas);
        free(alphas1);
        free(alphas2);
        free(y_hat1);
        free(y_hat2);
        sp1.Release();
        sp2.Release();
    } else {
        cout << "server" << MPI_rank << "..." << endl;
        double offline_etime;
        MPI_Status status;
        int flag;
        float y_hat[train_size];
        float* img = (float*)malloc(sizeof(float) * train_size * SIZE);
        float* alphas = (float*)malloc(sizeof(float) * train_size * OUT_SIZE);

        MPI_Recv(&flag, 1, MPI_INT, MPI_client, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(img, train_size * SIZE, MPI_FLOAT, MPI_client, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(alphas, train_size * OUT_SIZE, MPI_FLOAT, MPI_client, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(y_hat, train_size, MPI_FLOAT, MPI_client, 0, MPI_COMM_WORLD, &status);
        Triplet T1, T2, T3;
        batch_size = train_size;
        T1.GetShape(batch_size, SIZE, SIZE, 1);
        T1.Initial();
        T2.GetShape(OUT_SIZE, batch_size, batch_size, 1);
        T2.Initial();
        T3.GetShape(OUT_SIZE, SIZE, SIZE, OUT_SIZE);
        T3.Initial();
        float commTime = 0;
        T1.Recv(MPI_client);
        T2.Recv(MPI_client);
        T3.Recv(MPI_client);
        commTime += T1.commtime + T2.commtime + T3.commtime;
        offline_etime = timestamp();
        cout << "Comm time1: " << commTime << endl;
        cout << "Server" << MPI_rank << ": offline end time:" << offline_etime - offline_stime << "." << endl;
        double online_stime, online_etime;

        int MPI_dest;
        if (MPI_rank == MPI_server1) {
            MPI_dest = MPI_server2;
        } else {
            MPI_dest = MPI_server1;
        }

        float* batchY = y_hat;
        float* batchImg = img;
        online_stime = timestamp();
        float toler = 0.1;
        float C = 1;
        memset(alphas, 0, sizeof(float) * batch_size * OUT_SIZE);
        cout << batch_size << endl;
        cout << train_size << endl;
        for (int j = 0; j < 1; j++) {  //for i in range(m)
            float alphas_pro_Y[batch_size * OUT_SIZE];
            for (int k = 0; k < batch_size * OUT_SIZE; k++)
                alphas_pro_Y[k] = alphas[k] * batchY[k % batch_size];
            // T1.C = dataMatrix*dataMatrix[i,:].T

            T1.GetData(batchImg, batchImg + (j * SIZE));
            T1.Rec(MPI_dest);
            T1.OP(flag);
            // T2.C = fXi - b

            T2.GetData(alphas_pro_Y, T1.C);
            T2.Rec(MPI_dest);
            T2.OP(flag);
            commTime += T1.commtime + T2.commtime;
            float fXj = T2.C[0];
            float Ej = fXj - batchY[j];
            //float Ej = 0;

            if (((batchY[j] * Ej < -toler) && (alphas[j] < C)) || ((batchY[j] * Ej > toler) && (alphas[j] > 0))) {
                int l = rand() % (batch_size - 2);
                T1.GetData(batchImg, batchImg + (l * SIZE));
                T1.Rec(MPI_dest);
                T1.OP(flag);
                T2.GetData(alphas_pro_Y, T1.C);
                T2.Rec(MPI_dest);
                T2.OP(flag);
                float fXl = T2.C[0];
                float El = fXl - batchY[l];

                float alphaJold = alphas[j];
                float alphaLold = alphas[l];
                float L, H;
                if (batchY[j] != batchY[l]) {
                    L = MAX(0, alphas[l] - alphas[j]);
                    H = MIN(C, C + alphas[l] - alphas[j]);
                } else {
                    L = MAX(0, alphas[l] + alphas[j] - C);
                    H = MIN(C, alphas[l] + alphas[j]);
                }
                if (L == H)
                    continue;

                T3.GetData(batchImg + j * SIZE, batchImg + l * SIZE);
                T3.Rec(MPI_dest);
                //            T3.OP(flag);
                commTime += T3.commtime;
                float tmp1 = T3.C[0];
                T3.GetData(batchImg + j * SIZE, batchImg + j * SIZE);
                T3.Rec(MPI_dest);
                //              T3.OP(flag);
                commTime += T3.commtime;
                float tmp2 = T3.C[0];
                T3.GetData(batchImg + l * SIZE, batchImg + l * SIZE);
                T3.Rec(MPI_dest);
                //                T3.OP(flag);
                commTime += T3.commtime;
                float tmp3 = T3.C[0];
                float eta = 2 * tmp1 - tmp2 - tmp3;
                if (eta >= 0)
                    continue;
                alphas[l] -= batchY[l] * (Ej - El) / eta;
                alphas[l] = clipAlpha(alphas[l], H, L);
                if (alphas[l] - alphaLold < 0.0001 && alphas[l] - alphaLold > -0.0001)
                    continue;
                alphas[j] += batchY[l] * batchY[j] * (alphaLold - alphas[l]);
            }
            cout << "Communication time: " << commTime << endl;
        }

        online_etime = timestamp();
        cout << "Server" << MPI_rank << " train time:" << online_etime - online_stime << endl;
        cout << "Server" << MPI_rank << " send to client..." << endl;
        MPI_Send(alphas, train_size * OUT_SIZE, MPI_FLOAT, MPI_client, 0, MPI_COMM_WORLD);

        T1.Release();
        T2.Release();
        T3.Release();
        free(img);
        free(alphas);
        // free(y_hat);
    }
    MPI_Finalize();
    all_etime = timestamp();
    cout << "All time:" << all_etime - all_stime << endl;
    return 0;
}
