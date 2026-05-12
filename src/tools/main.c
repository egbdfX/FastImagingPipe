#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define VERB(name)                            \
    if(strcmp(argv[1], (#name)) == 0){        \
        extern int main_##name(int, char**);  \
        return main_##name(argc-1, argv+1);   \
    }


int main(int argc, char** argv){
    if(argc <= 1){
        fprintf(stderr, "Usage: %s <verb>", argv[0]);
        return EXIT_FAILURE;
    }

    VERB(pipe);

    fprintf(stderr, "Unknown verb \"%s\"", argv[1]);
    return EXIT_FAILURE;
}
