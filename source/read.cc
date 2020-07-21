#include <iostream>
#include <cstring>
#include "../include/read.h"
#include <fstream>
#include <ostream>
#include <istream>

void read_Images(string filename, float *images, int SIZE, int train_size){
    ifstream file(filename, ios::in);
    if(!file.is_open()){
        cout << "Open file failed." << endl;
    }
    for(int i = 0; i < train_size*SIZE; i++){
        file >> images[i];
    }
    file.close();
}
void read_Label(string filename, float *labels, int train_size){
    ifstream file(filename, ios::in);
    if(!file.is_open()){
        cout << "Open file failed." << endl;
    }
    for(int i = 0; i < train_size; i++){
        file >> labels[i];
    }
    file.close();
}
void Generator(float *array, int bound, int SIZE){
    for(int i = 0; i < SIZE; i++){
        array[i] = rand()%bound;
    }
}