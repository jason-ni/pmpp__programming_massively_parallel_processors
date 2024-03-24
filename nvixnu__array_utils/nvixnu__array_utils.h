#ifndef NVIXNU__ARRAY_UTILS_H_
#define NVIXNU__ARRAY_UTILS_H_

/**
* Receives an array and iterates through its elements
* @param v The array pointer
* @param el_size The size in bytes of one array element
* @param len The array lenght (The number of elements in the array)
* @param cb The callback that receive the element and its index in each iteration
*/
void nvixnu__array_map(void *v, size_t el_size, int len, void (*cb)(void *, int));

/**
* Prints an element of type double and a new line
* @param el The pointer to the element to be printed
* @param i The element index
*/
void nvixnu__print_item_double( void *el, int i);

/**
* Prints an element of type int and a new line
* @param el The pointer to the element to be printed
* @param i The element index
*/
void nvixnu__print_item_int( void *el, int i);

#endif /* NVIXNU__ARRAY_UTILS_H_ */
