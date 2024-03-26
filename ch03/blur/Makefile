CC = nvcc
OBJDIR = ./obj

CFLAGS = -g -G --compiler-options -Wall -lopencv_imgcodecs -lopencv_core

INCLUDES = $(patsubst %,-I ../../%, )
INCLUDES += -I /usr/include/opencv4

OBJS = $(patsubst %.cu,%.o,)

COMPILEDOBJS := $(patsubst %,$(OBJDIR)/%,$(notdir $(OBJS))) 

all: objdir $(OBJS)
	nvcc blur.cu -o blur.out $(COMPILEDOBJS) $(CFLAGS) $(INCLUDES)

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
	rm -Rf blur.out

.PHONY: all clean app run objdir