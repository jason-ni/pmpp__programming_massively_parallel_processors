CC = nvcc
OBJDIR = ./obj

CFLAGS = -g -G --compiler-options -Wall 

DEPSNAMES = \
nvixnu__populate_arrays_utils \
nvixnu__array_utils \
nvixnu__error_utils \

INCLUDES = $(patsubst %,-I ../../%, )
INCLUDES += $(patsubst %,-I ../../%, $(DEPSNAMES))

OBJS = $(patsubst %.cu,%.o,)

COMPILEDOBJS := $(patsubst %,$(OBJDIR)/%,$(notdir $(OBJS))) 

all: objdir $(OBJS)
	nvcc matrixmul.cu -o matrixmul.out $(COMPILEDOBJS) $(CFLAGS) $(INCLUDES)

# Creates the ./obj dir
objdir:
	mkdir	-p	$(OBJDIR)

# Compile a dependency
%.o: %.cu
	nvcc -c $< -o $(OBJDIR)/$(notdir $@) $(CFLAGS)

# Run the executable
run:
	./main

# Remove the generated artifacts
clean:
	rm -Rf $(OBJDIR)/*.o
	rm -Rf matrixmul.out

.PHONY: all clean app run objdir