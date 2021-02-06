srun -N 3 -n 3 --nodelist=gorgon[5-7] linear ../setup/CIFAR
srun -N 3 -n 3 --nodelist=gorgon[5-7] conv ../setup/CIFAR
srun -N 3 -n 3 --nodelist=gorgon[5-7] MLP ../setup/CIFAR
srun -N 3 -n 3 --nodelist=gorgon[5-7] linear ../setup/CIFAR
srun -N 3 -n 3 --nodelist=gorgon[5-7] logistic ../setup/CIFAR

