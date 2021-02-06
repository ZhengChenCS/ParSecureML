#srun -N 3 -n 3 --nodelist=gorgon[2-4] linear ../setup/VGG

#srun -N 3 -n 3 --nodelist=gorgon[2-4] linear ../setup/NIST

# srun -N 3 -n 3 --nodelist=gorgon[5-7] linear ../setup/SYN

srun -N 3 -n 3 --nodelist=gorgon[2-4] linear ../setup/MNIST

#srun -N 3 -n 3 --nodelist=gorgon[2-4] linear ../setup/CIFAR
