CUDA_INSTALL_PATH = /usr/local/cuda-9.0
MPICC = mpicxx
NVCC = $(CUDA_INSTALL_PATH)/bin/nvcc
CC = g++
CFLAGS = -std=c++11
TARGET = bin/runner
SRCDIR = source
BUILDDIR = build
TESTDIR = test
OBJECTS = $(BUILDDIR)/compressCommunicate.o $(BUILDDIR)/offline.o $(BUILDDIR)/online.o $(BUILDDIR)/read.o $(BUILDDIR)/cuda.o 
INC = -I../../common/inc
LIBS = -L$(CUDA_INSTALL_PATH)/lib64 -lcudart -lcurand -lcublas -lpthread
RM = rm
TESTLinear = bin/linear
TESTLinearOBJ = build/line.o
TESTLogistic = bin/logistic
TESTLogisticOBJ = build/logistic.o
TESTMLP = bin/MLP
TESTMLPOBJ = build/MLP.o
TESTConv = bin/conv
TESTConvOBJ = build/conv.o
TESTRNN = bin/RNN
TESTRNNOBJ = build/RNN.o
TESTSVM = bin/SVM
TESTSVMOBJ = build/SVM.o
TESTSVMo = bin/SVMo
TESTSVMoOBJ = build/SVMo.o


$(TARGET): $(OBJECTS) $(BUILDDIR)/main.o
	@echo "Linking..."
	$(MPICC) $^ $(LIBS) -o $(TARGET) -fopenmp
$(BUILDDIR)/main.o : $(SRCDIR)/main.cc
	@mkdir -p $(BUILDDIR)
	$(MPICC) $(CFLAGS) $(INC) -c -fopenmp $(SRCDIR)/main.cc -o $(BUILDDIR)/main.o
$(BUILDDIR)/compressCommunicate.o: $(SRCDIR)/compressCommunicate.cc
	@mkdir -p $(BUILDDIR)
	$(MPICC) $(CFLAGS) $(INC) -c -fopenmp $(SRCDIR)/compressCommunicate.cc -o $(BUILDDIR)/compressCommunicate.o
$(BUILDDIR)/offline.o: $(SRCDIR)/offline.cc
	@mkdir -p $(BUILDDIR)
	$(MPICC) $(CFLAGS) $(INC) -c -fopenmp $(SRCDIR)/offline.cc -o $(BUILDDIR)/offline.o
$(BUILDDIR)/online.o: $(SRCDIR)/online.cc
	@mkdir -p $(BUILDDIR)
	$(MPICC) $(CFLAGS) $(INC) -c -fopenmp $(SRCDIR)/online.cc -o $(BUILDDIR)/online.o
$(BUILDDIR)/read.o: $(SRCDIR)/read.cc
	@mkdir -p $(BUILDDIR)
	$(MPICC) $(CFLAGS) $(INC) -c -fopenmp $(SRCDIR)/read.cc -o $(BUILDDIR)/read.o
$(BUILDDIR)/cuda.o: $(SRCDIR)/cuda.cu
	@mkdir -p $(BUILDDIR)
	$(NVCC) $(CFLAGS) $(INC) -c $(SRCDIR)/cuda.cu -o $(BUILDDIR)/cuda.o
TestLinear: $(TESTDIR)/line.cc
	$(MPICC) $(CFLAGS) $(INC) -c -fopenmp $(TESTDIR)/line.cc -o $(BUILDDIR)/line.o
	$(MPICC) -fopenmp $(OBJECTS) $(TESTLinearOBJ) $(LIBS) -o $(TESTLinear)
TestLogistic: $(TESTDIR)/logistic.cc
	$(MPICC) $(CFLAGS) $(INC) -c -fopenmp $(TESTDIR)/logistic.cc -o $(BUILDDIR)/logistic.o
	$(MPICC) -fopenmp $(OBJECTS) $(TESTLogisticOBJ) $(LIBS) -o $(TESTLogistic)
TestMLP: $(TESTDIR)/MLP.cc
	$(MPICC) $(CFLAGS) $(INC) -c -fopenmp $(TESTDIR)/MLP.cc -o $(BUILDDIR)/MLP.o
	$(MPICC) -fopenmp $(OBJECTS) $(TESTMLPOBJ) $(LIBS) -o $(TESTMLP)
TestConv: $(TESTDIR)/conv.cc
	$(MPICC) $(CFLAGS) $(INC) -c -fopenmp $(TESTDIR)/conv.cc -o $(BUILDDIR)/conv.o
	$(MPICC) -fopenmp $(OBJECTS) $(TESTConvOBJ) $(LIBS) -o $(TESTConv)
TestSVM: $(TESTDIR)/SVM.cc
	$(MPICC) $(CFLAGS) $(INC) -c -fopenmp $(TESTDIR)/SVM.cc -o $(BUILDDIR)/SVM.o
	$(MPICC) -fopenmp $(OBJECTS) $(TESTSVMOBJ) $(LIBS) -o $(TESTSVM)
TestSVMo: $(TESTDIR)/SVM.cc
	$(MPICC) $(CFLAGS) $(INC) -c -fopenmp $(TESTDIR)/svm_origin.cc -o $(BUILDDIR)/SVMo.o
	$(MPICC) -fopenmp $(OBJECTS) $(TESTSVMoOBJ) $(LIBS) -o $(TESTSVMo)
TestRNN: $(TESTDIR)/RNN.cc
	$(MPICC) $(CFLAGS) $(INC) -c -fopenmp $(TESTDIR)/RNN.cc -o $(BUILDDIR)/RNN.o
	$(MPICC) -fopenmp $(OBJECTS) $(TESTRNNOBJ) $(LIBS) -o $(TESTRNN)

clean:
	@echo "clean..."
	$(RM) -r $(BUILDDIR) $(TARGET) $(TEST)
