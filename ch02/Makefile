CC = nvcc
OBJDIR = ./obj

CFLAGS = -g -G --compiler-options -Wall

INCLUDES = $(patsubst %,-I ../%,)

OBJS = $(patsubst %.cu,%.o,)

COMPILEDOBJS := $(patsubst %,$(OBJDIR)/%,$(notdir $(OBJS))) 

all: objdir $(OBJS)
	nvcc vadd.cu -o vec_add.out $(COMPILEDOBJS) $(CFLAGS) $(INCLUDES)

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
	rm -Rf vec_add.out

.PHONY: all clean app run objdir