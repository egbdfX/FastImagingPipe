#
# Declare the source file for every tool in FIP.
#
# Each "verb" should have its own main file.
#

fip_tools_src = files(
    'main-help.c',
    'main-image.c',
    'main-pipe.c',
)


#
# Build main.[ch] files on-the-fly here.
#

fip_verbs    = []
verbtabldc   = f'const fip_verb_entry_t fip_verb_table[]'
verbnamedc   = f''
verbnamedf   = f''
verbfuncdc   = f''
verbtabldf   = f''
foreach f : fip_tools_src
    if fs.stem(f).startswith('main-')
        v           = fs.stem(f).substring(5)
        fip_verbs  += [v]
        verbnamedc += f'extern char main_@v@_name[];\n'
        verbnamedf += f'char        main_@v@_name[] = "fip @v@";\n'
        verbfuncdc += f'extern int  main_@v@(int argc, char* argv[]);\n'
        verbtabldf += f'    {main_@v@_name, main_@v@},\n'
    endif
endforeach


main_h       = f'''/* Include Guard */
#ifndef SRC_TOOLS_MAIN_H
#define SRC_TOOLS_MAIN_H


/* Includes */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <popt.h>


/* Predefined codes for libpopt flags parsing */
#define  HELP_FLAG               1
#define  VERBOSE_FLAG            2
#define  QUIET_FLAG              3


/* Extern "C" Guard */
#ifdef __cplusplus
extern "C" {
#endif


struct fip_verb_entry_t{
    char*   name;
    int   (*func)(int argc, char** argv);
};

typedef struct fip_verb_entry_t fip_verb_entry_t;


@verbnamedc@
@verbfuncdc@
extern @verbtabldc@;


/* End Extern "C" and Include Guard */
#ifdef __cplusplus
}
#endif
#endif
'''

main_c       = f'''#include "main.h"

@verbnamedf@
@verbtabldc@ = {
@verbtabldf@    {NULL, 0},
};

int main(int argc, char** argv){
    const fip_verb_entry_t* v;

    if(argv[0] && argv[1])
        for(v=fip_verb_table; v->name; v++)
            if(strcmp(v->name+4, argv[1]) == 0)
                return argv[1]=v->name, v->func(argc-1, argv+1);

    return main_help(argc, argv);
}
'''

fip_tools_src += [
    configure_file(output: 'main.h', capture: true,
                   command: [py, '-c', 'import sys;print(sys.argv[1])', main_h]),
    configure_file(output: 'main.c', capture: true,
                   command: [py, '-c', 'import sys;print(sys.argv[1])', main_c]),
]

unset_variable('main_h')
unset_variable('main_c')
unset_variable('verbtabldc')
unset_variable('verbnamedc')
unset_variable('verbnamedf')
unset_variable('verbfuncdc')
unset_variable('verbtabldf')
