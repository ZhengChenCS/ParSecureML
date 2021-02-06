srun -N 3 -n 3 --nodelist=gorgon[2-4] MLP ../setup/VGG

srun -N 3 -n 3 --nodelist=gorgon[2-4] MLP ../setup/NIST

srun -N 3 -n 3 --nodelist=gorgon[2-4] MLP ../setup/SYN

srun -N 3 -n 3 --nodelist=gorgon[2-4] MLP ../setup/MNIST

srun -N 3 -n 3 --nodelist=gorgon[2-4] MLP ../setup/CIFAR
