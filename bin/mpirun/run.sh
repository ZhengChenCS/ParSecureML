mpirun -np 3  ../runner
mpirun -np 3  ../linear ../../setup/MNIST
mpirun -np 3 ../logistic ../../setup/MNIST
mpirun -np 3  ../MLP ../../setup/MNIST
mpirun -np 3 ../conv ../../setup/MNIST
mpirun -np 3  ../RNN
mpirun -np 3 ../SVM ../../setup/MNIST