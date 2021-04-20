NVCC=nvcc
GCC=g++

CONSTANT_INCLUDE=../constant.h
CUDA_INCLUDEPATH=/usr/local/cuda-11.2/include
NVCC_OPTS=-O3 -arch=sm_60 -Xcompiler -Wall -Xcompiler -Wextra -m64 -rdc=true -lcudadevrt -G

# Source file directory:
SRC_DIR = src

# Object file directory:
OBJ_DIR = bin

# Include header file diretory:
INC_DIR = include

# Target executable name:
EXE = ga

# Object files:
OBJS = $(OBJ_DIR)/evaluation.o $(OBJ_DIR)/mutation.o $(OBJ_DIR)/population.o $(OBJ_DIR)/crossover.o $(OBJ_DIR)/migration.o $(OBJ_DIR)/selection.o $(OBJ_DIR)/GA.o

$(EXE): $(OBJS) Makefile
	$(NVCC) -o $@ $(OBJS) -L $(NVCC_OPTS)

# Compile main
$(OBJ_DIR)/%.o: %.cu
	$(NVCC) -c $< -o $@ $(NVCC_OPTS)

# Compile C++ Source files to object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp include/%.h
	$(GCC) -c $< -o $@

# Compile CUDA source files ot object files
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu $(INC_DIR)/%.cuh $(CONSTANT_INCLUDE)
	$(NVCC) -c $< -o $@ $(NVCC_OPTS)

clean:
	$(RM) bin/* *.o $(EXE)