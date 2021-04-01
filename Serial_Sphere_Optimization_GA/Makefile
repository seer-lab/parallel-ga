GCC=g++

# Source file directory:
SRC_DIR = src

# Object file directory:
OBJ_DIR = bin

# Include header file diretory:
INC_DIR = include

# Target executable name:
EXE = ga

# Object files:
OBJS = $(OBJ_DIR)/evaluation.o $(OBJ_DIR)/mutation.o $(OBJ_DIR)/population.o $(OBJ_DIR)/crossover.o $(OBJ_DIR)/replacement.o $(OBJ_DIR)/selection.o $(OBJ_DIR)/GA.o

$(EXE): $(OBJS) Makefile
	$(GCC) -o $@ $(OBJS) 

# Compile main
$(OBJ_DIR)/%.o: %.cpp
	$(GCC) -c $< -o $@ 

# Compile C++ Source files to object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp include/%.h
	$(GCC) -c $< -o $@

clean:
	$(RM) bin/* *.o $(EXE)