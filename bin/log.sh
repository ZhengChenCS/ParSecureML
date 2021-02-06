srun -N 3 -n 3 --nodelist=gorgon[5-7] logistic ../setup/VGG

srun -N 3 -n 3 --nodelist=gorgon[5-7] logistic ../setup/NIST

srun -N 3 -n 3 --nodelist=gorgon[5-7] logistic ../setup/SYN

srun -N 3 -n 3 --nodelist=gorgon[5-7] logistic ../setup/MNIST

srun -N 3 -n 3 --nodelist=gorgon[5-7] logistic ../setup/CIFAR
