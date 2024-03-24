#include "nvixnu__populate_arrays_utils.h"
#include <stdio.h>
#include <stdarg.h>
#include <string.h>


void nvixnu__populate_array_from_file(const char * path, const char * pattern, int length, size_t el_size, void *array){
  FILE * fp;
  
  fp = fopen (path, "r");

  char *u = (char *)array;
  for(int i = 0; i < length; i++){
	fscanf(fp, pattern, u + i*el_size); 
  }
  
  fclose(fp);
}

void nvixnu__populate_multiple_arrays_from_file(const char * path, const char * discard_prev_asterisk, const char * pattern, const char * discard_post_asterisk, int va_list_array_length, size_t el_size, int va_list_size, ...){
  FILE * fp;
  int i, j, read_prev, read_post;
  void **args_ptrs;
  va_list args;
  va_start(args, va_list_size);
  
  //Reads the va_list to args_ptrs
  args_ptrs = (void** )malloc(va_list_size*el_size*sizeof(void *));
  for(i = 0; i < va_list_size; i++){
      args_ptrs[i] = va_arg(args, void *);
  }
  va_end(args);

  fp = fopen (path, "r");

  read_prev = strcmp(discard_prev_asterisk, "");
  read_post = strcmp(discard_post_asterisk, "");

  for(i = 0; i < va_list_array_length; i++){ //iterates through each array position
      for(j = 0; j < va_list_size; j++){ //iterates through each array that will be populated
          if(read_prev){
            fscanf(fp, discard_prev_asterisk); // discards the previous values that doesn't matters
          }
          fscanf(fp, pattern, args_ptrs[j] +i*el_size); // reads the values that matters
          if(read_post){
            fscanf(fp, discard_post_asterisk); // discards the subsequent values that doesn't matters
          }
      }
  }
  
  free(args_ptrs);
  fclose(fp);
}
