srun -N 3 -n 3 --nodelist=gorgon[5-7] conv ../setup/VGG

srun -N 3 -n 3 --nodelist=gorgon[5-7] conv ../setup/NIST

srun -N 3 -n 3 --nodelist=gorgon[5-7] conv ../setup/SYN

srun -N 3 -n 3 --nodelist=gorgon[5-7] conv ../setup/MNIST

srun -N 3 -n 3 --nodelist=gorgon[5-7] conv ../setup/CIFAR
