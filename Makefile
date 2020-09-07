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


$(TARGET): $(OBJECTS) $(BUILDDIR)/main.o
	@echo "Linking..."
	$(MPICC) $^ $(LIBS) -o $(TARGET)
$(BUILDDIR)/main.o : $(SRCDIR)/main.cc
	@mkdir -p $(BUILDDIR)
	$(MPICC) $(CFLAGS) $(INC) -c $(SRCDIR)/main.cc -o $(BUILDDIR)/main.o
$(BUILDDIR)/compressCommunicate.o: $(SRCDIR)/compressCommunicate.cc
	@mkdir -p $(BUILDDIR)
	$(MPICC) $(CFLAGS) $(INC) -c $(SRCDIR)/compressCommunicate.cc -o $(BUILDDIR)/compressCommunicate.o
$(BUILDDIR)/offline.o: $(SRCDIR)/offline.cc
	@mkdir -p $(BUILDDIR)
	$(MPICC) $(CFLAGS) $(INC) -c $(SRCDIR)/offline.cc -o $(BUILDDIR)/offline.o
$(BUILDDIR)/online.o: $(SRCDIR)/online.cc
	@mkdir -p $(BUILDDIR)
	$(MPICC) $(CFLAGS) $(INC) -c $(SRCDIR)/online.cc -o $(BUILDDIR)/online.o
$(BUILDDIR)/read.o: $(SRCDIR)/read.cc
	@mkdir -p $(BUILDDIR)
	$(MPICC) $(CFLAGS) $(INC) -c $(SRCDIR)/read.cc -o $(BUILDDIR)/read.o
$(BUILDDIR)/cuda.o: $(SRCDIR)/cuda.cu
	@mkdir -p $(BUILDDIR)
	$(NVCC) $(CFLAGS) $(INC) -c $(SRCDIR)/cuda.cu -o $(BUILDDIR)/cuda.o
TestLinear: $(TESTDIR)/line.cc
	$(MPICC) $(CFLAGS) $(INC) -c $(TESTDIR)/line.cc -o $(BUILDDIR)/line.o
	$(MPICC) $(OBJECTS) $(TESTLinearOBJ) $(LIBS) -o $(TESTLinear)

clean:
	@echo "clean..."
	$(RM) -r $(BUILDDIR) $(TARGET) $(TEST)
