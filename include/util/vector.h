typedef struct {
  char ** values;
  int capacity;
  int size;
} StringVector;


int StringVector_init(StringVector *);

int StringVector_destroy(StringVector *);

int StringVector_size(StringVector *);

const char * StringVector_get(StringVector *, int index);

int StringVector_push(StringVector *, char * value);
