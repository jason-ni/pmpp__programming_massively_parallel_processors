#ifndef NVIXNU__POPULATE_ARRAYS_UTILS_H_
#define NVIXNU__POPULATE_ARRAYS_UTILS_H_

/**
* Reads the content of a file, line by line and put it into arrays (Double). Each element is read by the specified "pattern" and the undesirable values are discarded by the "discard_prev_asterisk" and "discard_post_asterisk" patterns.
* @param path The path of the file to be read
* @param discard_prev_asterisk The pattern of values to be discarded before the reading (Use * to ignore the read value. Ex: "%*d", "%*lf")
* @param pattern The pattern to be read from file
* @param discard_post_asterisk The pattern of values to be discarded after the reading (Use * to ignore the read value. Ex: "%*d", "%*lf")
* @param va_list_array_length The length of the arrays to be populated
* @param el_size The size of each element
* @param va_list_size The number of the arrays to be populated (The number of pointers in the next parameters (va_list))
* @param {va_list} ... The pointers to arrays to be populated
*/
void nvixnu__populate_multiple_arrays_from_file(const char * path, const char * discard_prev_asterisk, const char * pattern, const char * discard_post_asterisk, int va_list_array_length, size_t el_size, int va_list_size, ...);

/**
* Reads the content of a file, line by line and put it into an array. Each element is read by the specified "pattern".
* @param path The path of the file to be read
* @param pattern The pattern to be read from file
* @param length The length of the array to be populated
* @param el_size The size of each element
* @param array The array to be populated
*/
void nvixnu__populate_array_from_file(const char * path, const char * pattern, int length, size_t el_size, void *array);
#endif /* NVIXNU__POPULATE_ARRAYS_UTILS_H_ */
