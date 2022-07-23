#include "../../include/util/macros.h"
#include "../../include/util/vector.h"
#include <stdlib.h>
#include <string.h>


#define INIT_CAP    8
#define SCALE_CAP   2


int StringVector_init(StringVector * v) {
    v->values = MALLOC(char *, INIT_CAP);
    v->capacity = INIT_CAP;
    v->size = 0;
    return 0;
}

int StringVector_destroy(StringVector * v) {
    for(int i = 0; i < v->size; i++) {
        FREE(v->values[i]);
    }
    FREE(v->values);
    return 0;
}

int StringVector_size(StringVector * v) {
    return v->size;
}

const char * StringVector_get(StringVector * v, int index) {
    return v->values[index];
}

static void grow_if_need(StringVector * v) {
    if(v->size == v->capacity) {
        v->capacity *= SCALE_CAP;
        v->values = REALLOC(v->values, char *, v->capacity);
    }
}

int StringVector_push(StringVector * v, char * value) {
    grow_if_need(v);
    v->values[v->size] = strdup(value);
    v->size++;
    return 0;
}
