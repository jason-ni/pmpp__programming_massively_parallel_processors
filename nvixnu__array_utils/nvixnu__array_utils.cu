#include "nvixnu__array_utils.h"
#include <stdio.h>


void nvixnu__array_map(void *v, size_t el_size, int len, void (*cb)(void *, int)){
  int i;
  char *u = (char *)v;
  for (i = 0; i < len; i++) {
    cb(u+i*el_size, i);
  }    
}

void nvixnu__print_item_double( void *el, int i){
    printf("%06d\t%f \n", i, *((double *)el));
}

void nvixnu__print_item_int( void *el, int i){
    printf("%06d\t%d \n", i, *((int *)el));
}
