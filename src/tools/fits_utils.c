/* Includes */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>

#include "fits_utils.h"



/* Defines */
#define FITS_RETURN(ret)                        \
    do{                                         \
        return status ? *status=(ret) : (ret);  \
    }while(0)

#define FITS_CHECKED(x)                         \
    do{                                         \
        int _statustmp = (x);                   \
        if(_statustmp){                         \
            FITS_RETURN((_statustmp));          \
        }                                       \
    }while(0)

#define FITS_ASSERT(expr, ret, ...)             \
    do{                                         \
        if(!(expr)){                            \
            fprintf(stderr, __VA_ARGS__);       \
            FITS_RETURN((ret));                 \
        }                                       \
    }while(0)


ssize_t fip_pwrite_fully(int fd, const void* buf, size_t len, off_t off){
    ssize_t ret=0, tot=0;
    while(len){
        ret = pwrite(fd, buf, len, off);
        if(ret <= 0){
            return tot>0 ? tot : ret;
        }else{
            buf  = (const void*)((const char*)buf + ret);
            len -= ret;
            off += ret;
            tot += ret;
        }
    }
    return tot;
}

size_t fip_compute_missing_records(size_t datastart){
    return (datastart >> 6) * -37U & 63;
}

int fip_input_open_diskfile(fitsfile** fptr, const char* filename, int iomode, int* status){
    int       exttype    =  0;
    int       naxis      =  0;
    long long visaxis[3] = {0};
    long long binaxis[3] = {0};
    long long vinaxis[3] = {0};
    int       visdtype   =  0;
    int       bindtype   =  0;
    long long visdstart  = -1;
    long long bindstart  = -1;


    /* Open File */
    FITS_CHECKED(fits_open_diskfile (fptr,     filename,   iomode,  status));
    FITS_CHECKED(fits_movabs_hdu    (*fptr,    1,  &exttype,        status));


    /* Primary HDU (index=1 in 1-based indexing) is ignored. */


    /* First Extension HDU (index=2) must be the transform matrices. */
    FITS_CHECKED(fits_movrel_hdu    (*fptr,   1,  &exttype,         status));
    FITS_ASSERT (exttype == IMAGE_HDU,     NOT_IMAGE,
        "HDU 2 (transform matrices) of file %s is not an IMAGE_HDU, "
        "thus cannot be a properly constructed FIP input file!\n",
        filename);
    FITS_CHECKED(fits_get_img_dim   (*fptr,       &naxis,           status));
    FITS_ASSERT (naxis == 3,               BAD_NAXIS,
        "HDU 2 (transform matrices) of file %s is supposed to be shaped "
        "{num_snapshots, 3, 3} (in C-order), but NAXIS was %d, thus "
        "cannot be a properly constructed FIP input file!\n",
        filename, naxis);
    FITS_CHECKED(fits_get_img_sizell(*fptr,   3,   vinaxis,         status));
    FITS_ASSERT (vinaxis[0] == 3,          BAD_NAXES,
        "HDU 2 (transform matrices) of file %s is supposed to be shaped "
        "{num_snapshots, 3, 3} (in C-order), but NAXIS1 was %lld, "
        "thus cannot be a properly constructed FIP input file!\n",
        filename, vinaxis[0]);
    FITS_ASSERT (vinaxis[1] == 3,          BAD_NAXES,
        "HDU 2 (transform matrices) of file %s is supposed to be shaped "
        "{num_snapshots, 3, 3} (in C-order), but NAXIS2 was %lld, "
        "thus cannot be a properly constructed FIP input file!\n",
        filename, vinaxis[1]);


    /* Second Extension HDU (index=3) must be the visibilities. */
    FITS_CHECKED(fits_movrel_hdu    (*fptr,   1,  &exttype,         status));
    FITS_ASSERT (exttype == IMAGE_HDU, NOT_IMAGE,
        "HDU 3 (visibilities) of file %s is not an IMAGE_HDU, "
        "thus cannot be a properly constructed FIP input file!\n",
        filename);
    FITS_CHECKED(fits_get_img_dim   (*fptr,       &naxis,           status));
    FITS_ASSERT (naxis == 3,           BAD_NAXIS,
        "HDU 3 (visibilities) of file %s is supposed to be shaped "
        "{num_snapshots, num_baselines, 2} (in C-order), but NAXIS "
        "was %d, thus cannot be a properly constructed FIP input file!\n",
        filename, naxis);
    FITS_CHECKED(fits_get_img_sizell(*fptr,   3,   visaxis,         status));
    FITS_ASSERT (visaxis[0] == 2,          BAD_NAXES,
        "HDU 3 (visibilities) of file %s is supposed to be shaped "
        "{num_snapshots, num_baselines, 2} (in C-order), but NAXIS1 "
        "was %lld, thus cannot be a properly constructed FIP input file!\n",
        filename, visaxis[0]);
    FITS_ASSERT (vinaxis[2] == visaxis[2], BAD_NAXES,
        "HDU 3 (visibilities) of file %s is supposed to match HDU 2 "
        "(transform matrices) in num_snapshots, but their NAXIS3 values "
        "disagree (%lld != %lld); thus this file cannot be a properly "
        "constructed FIP input file!\n",
        filename, vinaxis[2], visaxis[2]);
    FITS_CHECKED(fits_get_img_type  (*fptr,       &visdtype,        status));
    FITS_ASSERT (visdtype == FLOAT_IMG,    BAD_BITPIX,
        "HDU 3 (visibilities) of file %s is supposed to be FLOAT32, "
        "but is not; thus this file cannot be a properly constructed "
        "FIP input file!\n",
        filename);
    FITS_CHECKED(fits_get_hduaddrll (*fptr, NULL, &visdstart, NULL, status));
    FITS_ASSERT (!fip_compute_missing_records(visdstart), BAD_HEADER_FILL,
        "HDU 3 (visibilities) of file %s is missing %zu records of 2880 "
        "bytes in its header; thus this file cannot be a properly "
        "constructed FIP input file!\n",
        filename, fip_compute_missing_records(visdstart));


    /* Third Extension HDU (index=4) must be the (r-)coordinates. */
    FITS_CHECKED(fits_movrel_hdu    (*fptr,   1,  &exttype,         status));
    FITS_ASSERT (exttype == IMAGE_HDU,     NOT_IMAGE,
        "HDU 4 (r-coordinates) of file %s is not an IMAGE_HDU, "
        "thus cannot be a properly constructed FIP input file!\n",
        filename);
    FITS_CHECKED(fits_get_img_dim   (*fptr,       &naxis,           status));
    FITS_ASSERT (naxis == 3,               BAD_NAXIS,
        "HDU 4 (r-coordinates) of file %s is supposed to be shaped "
        "{num_snapshots, num_baselines, 2} (in C-order), but NAXIS "
        "was %d, thus cannot be a properly constructed FIP input file!\n",
        filename, naxis);
    FITS_CHECKED(fits_get_img_sizell(*fptr,   3,   binaxis,         status));
    FITS_ASSERT (binaxis[0] == 2,          BAD_NAXES,
        "HDU 4 (r-coordinates) of file %s is supposed to be shaped "
        "{num_snapshots, num_baselines, 2} (in C-order), but NAXIS1 "
        "was %lld, thus cannot be a properly constructed FIP input file!\n",
        filename, binaxis[0]);
    FITS_ASSERT (binaxis[1] == visaxis[1], BAD_NAXES,
        "HDU 4 (r-coordinates) of file %s is supposed to match HDU 3 "
        "(visibilities) in num_baselines, but their NAXIS2 values disagree "
        "(%lld != %lld); thus this file cannot be a properly constructed "
        "FIP input file!\n",
        filename, binaxis[1], visaxis[1]);
    FITS_ASSERT (binaxis[2] == visaxis[2], BAD_NAXES,
        "HDU 4 (r-coordinates) of file %s is supposed to match HDU 2 "
        "(visibilities) in num_snapshots, but their NAXIS3 values disagree "
        "(%lld != %lld); thus this file cannot be a properly constructed "
        "FIP input file!\n",
        filename, binaxis[2], visaxis[2]);
    FITS_CHECKED(fits_get_img_type  (*fptr,       &bindtype,        status));
    FITS_ASSERT (bindtype == FLOAT_IMG,    BAD_BITPIX,
        "HDU 4 (r-coordinates) of file %s is supposed to be FLOAT32, "
        "but is not; thus this file cannot be a properly constructed "
        "FIP input file!\n",
        filename);
    FITS_CHECKED(fits_get_hduaddrll (*fptr, NULL, &bindstart, NULL, status));
    FITS_ASSERT (!fip_compute_missing_records(bindstart), BAD_HEADER_FILL,
        "HDU 4 (r-coordinates) of file %s is missing %zu records of 2880 "
        "bytes in its header; thus this file cannot be a properly "
        "constructed FIP input file!\n",
        filename, fip_compute_missing_records(bindstart));


    /* Rewind to beginning of file */
    FITS_CHECKED(fits_movabs_hdu    (*fptr,   1,  &exttype,         status));


    /* Exit */
    FITS_RETURN(0);
}

int fip_input_get_stats(fitsfile *fptr, long long* num_snapshots, long long* num_baselines, int* status){
    long long visaxis[3] = {0};

    *num_snapshots = 0;
    *num_baselines = 0;

    /**
     * Consult HDU 3 (visibilities), which should have
     *     NAXIS=3
     *     NAXIS1=2
     *     NAXIS2=num_baselines
     *     NAXIS3=num_snapshots
     * Then rewind to beginning of file.
     */

    FITS_CHECKED(fits_movabs_hdu    (fptr, 3, NULL,    status));
    FITS_CHECKED(fits_get_img_sizell(fptr, 3, visaxis, status));
    FITS_CHECKED(fits_movabs_hdu    (fptr, 1, NULL,    status));

    if(visaxis[0] != 2 || visaxis[1] <= 0 || visaxis[2] <= 0)
        FITS_RETURN(BAD_NAXES);

    *num_baselines = visaxis[1];
    *num_snapshots = visaxis[2];


    /* Exit */
    FITS_RETURN(0);
}

int fip_output_validate_diskfile(fitsfile*   fptr,
                                 const char* filename,
                                 long long   snap_count,
                                 long long   unit_num,
                                 int*        status){
    int       exttype    =  0;
    int       naxis      =  0;
    long long outaxis[3] = {0};
    int       outdtype   =  0;
    long long outdstart  = -1;


    /* Move to primary HDU */
    FITS_CHECKED(fits_movabs_hdu    (fptr,    1, &exttype,         status));


    /* Primary HDU (index=1 in 1-based indexing) must be the output image array. */
    FITS_ASSERT (exttype == IMAGE_HDU,     NOT_IMAGE,
        "HDU 1 (output image array) of file %s is not an IMAGE_HDU, "
        "thus cannot be a properly constructed FIP output file!\n",
        filename);
    FITS_CHECKED(fits_get_img_dim   (fptr,       &naxis,           status));
    FITS_ASSERT (naxis == 3,               BAD_NAXIS,
        "HDU 1 (output image array) of file %s is supposed to be shaped "
        "{num_snapshots, unit_num, unit_num} (in C-order), but NAXIS was %d, "
        "thus cannot be a properly constructed FIP output file!\n",
        filename, naxis);
    FITS_CHECKED(fits_get_img_sizell(fptr,   3,   outaxis,         status));
    FITS_ASSERT (outaxis[0] == unit_num,   BAD_NAXES,
        "HDU 1 (output image array) of file %s is supposed to be shaped "
        "{num_snapshots, unit_num, unit_num} (in C-order), but NAXIS1 was %lld, "
        "thus cannot be a properly constructed FIP output file!\n",
        filename, outaxis[0]);
    FITS_ASSERT (outaxis[1] == unit_num,   BAD_NAXES,
        "HDU 1 (output image array) of file %s is supposed to be shaped "
        "{num_snapshots, unit_num, unit_num} (in C-order), but NAXIS2 was %lld, "
        "thus cannot be a properly constructed FIP output file!\n",
        filename, outaxis[1]);
    FITS_ASSERT (outaxis[2] == snap_count, BAD_NAXES,
        "HDU 1 (output image array) of file %s is supposed to be shaped "
        "{num_snapshots, unit_num, unit_num} (in C-order), but NAXIS3 was %lld, "
        "thus cannot be a properly constructed FIP output file!\n",
        filename, outaxis[2]);
    FITS_CHECKED(fits_get_img_type  (fptr,       &outdtype,        status));
    FITS_ASSERT (outdtype == FLOAT_IMG,    BAD_BITPIX,
        "HDU 1 (output image array) of file %s is supposed to be FLOAT32, "
        "but is not; thus this file cannot be a properly constructed "
        "FIP output file!\n",
        filename);
    FITS_CHECKED(fits_get_hduaddrll (fptr, NULL, &outdstart, NULL, status));
    FITS_ASSERT (!fip_compute_missing_records(outdstart), BAD_HEADER_FILL,
        "HDU 1 (output image array) of file %s is missing %zu records of 2880 "
        "bytes in its header; thus this file cannot be a properly "
        "constructed FIP output file!\n",
        filename, fip_compute_missing_records(outdstart));


    /* Exit */
    FITS_RETURN(0);
}

/**
 * @brief Write header to FIP output file.
 *
 * Similar interface to fits_create_diskfile().
 *
 * @param [out]  fptr         FITS file pointer.
 * @param [in]   filename     Path to file to open.
 * @param [out]  snap_count   Expected number of snapshots.
 * @param [out]  unit_num     Expected number of units per image.
 * @param [out]  status       FITS status code return.
 * @return 0 if successful, !0 otherwise.
 */

int fip_output_write_header(fitsfile**  fptr,
                            const char* filename,
                            long long   snap_count,
                            long long   unit_num,
                            int*        status){
    long long  outaxis[3] = {unit_num, unit_num, snap_count};
    long long  outdstart  = -1;
    int        hdrlen     = 0;
    int        hdrextra;


    /* Primary HDU (index=1 in 1-based indexing) must be the output image array. */
    FITS_CHECKED(fits_create_imgll   (*fptr, FLOAT_IMG, 3, outaxis,  status));
    FITS_CHECKED(fits_get_hdrpos     (*fptr, &hdrlen,   NULL,        status));
    hdrextra = 36*64 - hdrlen%(36*64) - 1;
    FITS_CHECKED(fits_set_hdrsize    (*fptr, hdrextra,               status));
    FITS_CHECKED(fits_get_hduaddrll  (*fptr, NULL, &outdstart, NULL, status));
    FITS_ASSERT (!fip_compute_missing_records(outdstart), BAD_HEADER_FILL,
        "HDU 1 (output image array) of file %s is missing %zu records of 2880 "
        "bytes in its header; thus this file cannot be a properly "
        "constructed FIP output file!\n",
        filename, fip_compute_missing_records(outdstart));


    /* Exit */
    FITS_RETURN(0);
}

int fip_output_create_diskfile(fitsfile**  fptr,
                               const char* filename,
                               long long   snap_count,
                               long long   unit_num,
                               int*        status){
    FITS_CHECKED(fits_create_diskfile(fptr, filename,                       status));
    return fip_output_write_header   (fptr, filename, snap_count, unit_num, status);
}

int fip_output_open_diskfile  (fitsfile**  fptr,
                               const char* filename,
                               long long   snap_count,
                               long long   unit_num,
                               int*        status){
    return fip_output_openat_diskfile(fptr, AT_FDCWD, filename,
                                      snap_count, unit_num, status);
}

int fip_output_openat_diskfile(fitsfile**  fptr,
                               int         dirfd,
                               const char* filename,
                               long long   snap_count,
                               long long   unit_num,
                               int*        status){
    char SLASH[] = "/";
    char DOT[]   = ".";
#if PATH_MAX > 65536
    char p[PATH_MAX];
#else
    char p[65536];
#endif
    char procpath[sizeof("/proc/self/fd/1098765432109876543210")];
    char HDR[64*2880];
    const char* f;
    char* d;
    int  fd=-1, internaldfd=-1, len, hdrlen, rc, mode=0644, safe=1;
    struct stat fd_stat;
    LONGLONG outstart=sizeof(HDR), outdend=0;


    /* Fail with invalid output if error status already set. */
    if(*status)
        return -EINVAL;


    /* Initialize fptr to NULL. */
    *fptr = NULL;


    /* Output path cannot be NULL or empty */
    if(!filename || !*filename){
        *status = FILE_NOT_OPENED;
        return -EINVAL;
    }


    /* Output path cannot be a directory */
    if(!strcmp(filename, "/") ||
       !strcmp(filename, ".") ||
       !strcmp(filename, "..")){
        *status = FILE_NOT_OPENED;
        return -EISDIR;
    }


    /* Measure length of path */
    len = strlen(filename);


    /* Split path */
    for(f=filename+len; --f>filename;)
        if(*f == '/')
            break;
    if(*f == '/')
        f++;


    /* Check final path segment */
    if((f[0]=='\0') ||
       (f[0]=='.' && f[1]=='\0') ||
       (f[0]=='.' && f[1]=='.'  && f[2]=='\0')){
        *status = FILE_NOT_OPENED;
        return -EISDIR;
    }else if(f == filename){
        d = DOT;
    }else if(f == filename+1){
        d = SLASH;
    }else if((size_t)(f-filename) < sizeof(p)){
        d = memcpy(p, filename, f-filename);
        d[f-filename] = '\0';
    }else{
        *status = FILE_NOT_OPENED;
        return -ENAMETOOLONG;
    }


    /* Filename now split; Open FD for parent directory of file, if needed */
    if(f > filename){
        dirfd = internaldfd = openat(dirfd, d, O_NOCTTY|O_CLOEXEC|O_DIRECTORY|O_RDONLY
#ifdef _GNU_SOURCE
                                              |O_PATH
#endif
        );
        rc = errno;
        if(dirfd < 0)
            goto fatal_fits_not_opened;
    }


    /* Attempt opening file itself */
    retry_openat_existing:
    fd = openat(dirfd, f, O_NOCTTY|O_CLOEXEC|O_RDWR);
    rc = errno;
    if(fd >= 0){
        /**
         * The file pre-existed on the filesystem, and we successfully
         * opened it. Try using it by referencing
         *
         *     /proc/self/fd/<fd>
         *
         * If the kernel supports safe creation of preformed files with
         * O_TMPFILE, there will be no race.
         */

        close(internaldfd);
        snprintf(procpath, sizeof(procpath), "/proc/self/fd/%ld", (long)fd);
        if(fits_open_diskfile           (fptr, procpath, READWRITE,            status) ||
           fip_output_validate_diskfile(*fptr, procpath, snap_count, unit_num, status)){
            if(*fptr)
                fits_close_file(*fptr, status);
            *fptr = NULL;
            close(fd);
            return -EINVAL;
        }

        /* Success! */
        return fd;
    }else{
        /**
         * The file did not already exist, or could not be opened. There are
         * many possible reasons for this. If the file could not be opened for
         * reasons other than not existing, abort; There is no sense in
         * continuing.
         *
         * But if the file simply did not exist already, then a race has now
         * opened with other processes that might try creating it for us.
         *
         * Be careful!
         */

        if(rc != ENOENT)
            goto fatal_fits_not_opened;


        /**
         * Attempt to open an O_TMPFILE safe temporary file.
         *
         * If the system is compile-time incapable of O_TMPFILE, emulate a
         * failure by (-1, EOPNOTSUPP) and move directly to less safe file
         * creation.
         */

#ifdef O_TMPFILE
        fd = openat(dirfd, DOT, O_NOCTTY|O_CLOEXEC|O_RDWR|O_TMPFILE, mode);
#else
        fd = -1; errno = EOPNOTSUPP;
#endif
        rc = errno;
        if(fd < 0){
            if(rc == EOPNOTSUPP){
                fd = openat(dirfd, f, O_NOCTTY|O_CLOEXEC|O_RDWR|O_CREAT|O_EXCL, mode);
                rc = errno;
                if(fd < 0){
                    if(rc == EEXIST)
                        /**
                         * We lost the race, and the file was created by someone else.
                         * Moreover, if we get here then O_TMPFILE is not supported, therefore
                         * the race condition cannot be palliated.
                         *
                         * Proceeding, but at high danger!
                         */

                        goto retry_openat_existing;
                }else{
                    safe = 0;
                }
            }
        }
    }


    /* If we could not create the file any which way, crash out. */
    if(fd < 0)
        goto fatal_fits_not_created;


    /**
     * Write to target file, filling it with valid header.
     *
     * Because of difficulties with CFITSIO, synthesize our own header, obeying
     * our own precise requirements exactly.
     */

    snprintf(procpath, sizeof(procpath), "/proc/self/fd/%ld", (long)fd);
    hdrlen = snprintf(HDR, sizeof(HDR),
        "SIMPLE  =                    T / file does conform to FITS standard             "
        "BITPIX  =                  -32 / number of bits per data pixel                  "
        "NAXIS   =                    3 / number of data axes                            "
        "NAXIS1  = %20lld / length of data axis 1                          "
        "NAXIS2  = %20lld / length of data axis 2                          "
        "NAXIS3  = %20lld / length of data axis 3                          "
        "EXTEND  =                    T / FITS dataset may contain extensions            "
        "COMMENT   FITS (Flexible Image Transport System) format is defined in 'Astronomy"
        "COMMENT   and Astrophysics', volume 376, page 359; bibcode: 2001A&A...376..359H ",
        unit_num,
        unit_num,
        snap_count
    );
    memset(HDR+hdrlen, ' ', sizeof(HDR)-hdrlen);
    memcpy(HDR+sizeof(HDR)-80, "END", 3);
    outdend  = sizeof(HDR) + sizeof(float)*unit_num*unit_num*snap_count;
    outdend += outdend%2880 == 0 ? 0 : 2880-(outdend%2880);
    if(fip_pwrite_fully(fd, HDR, sizeof(HDR), 0) < 0){
        rc = errno;
        goto fatal_fits_not_opened;
    }


    /**
     * Resize the file immediately to final size with special system
     * calls and cross fingers that they work.
     *
     * 1. First try fallocate(fd, 0, 0, len), which guarantees an atomic resize
     *    to *at least* the given size, and does not shrink the file if it is
     *    already larger for some reason.
     *
     * 2. If that does not work, try ftruncate(fd, len). This has the slight
     *    risk of truncating away HDUs beyond the Primary HDU of the file if
     *    such exist, but currently none are specified. Try to defend against
     *    this small truncation risk by checking the file size with fstat().
     *    If attempted, the size increase (or decrease) should be atomic.
     *
     * 3. If even that does not work, try posix_fallocate(), but only if safe
     *    to do so. posix_fallocate() uses fallocate() under the hood but will
     *    fall back to an emulation layer that fills the file manually with
     *    zeros. This is racy if multiple processes could be writing to the
     *    file trying to grow it. Thus, we only try this option if we have
     *    created an O_TMPFILE.
     */

    if((
#ifdef _GNU_SOURCE
        fallocate(fd, 0, outstart, outdend-outstart)
#else
        ((errno = ENOSYS), -1)
#endif
        ) < 0){
        rc = errno;
        switch(rc){
            case EBADF:
            case EFBIG:
            case EINVAL:
            case EIO:
            case ENOSPC:
            case EPERM:
            case EROFS:
            case ETXTBSY:
                goto fatal_fits_not_opened;
            case ENOSYS:
            case EOPNOTSUPP:
            default:
                /* Query file size. If this fails, fatal out. */
                if(fstat(fd, &fd_stat) < 0){
                    rc = errno;
                    goto fatal_fits_not_opened;
                }

                /* If the file is already as big as needed, break out. */
                if((LONGLONG)fd_stat.st_size >= outdend)
                    break;

                /* Else continue with resize attempts, first with ftruncate(). */
#if _XOPEN_SOURCE   >= 500     || \
    _POSIX_C_SOURCE >= 200112L || \
    (_BSD_SOURCE - 0)
                if(ftruncate(fd, outdend) == 0)
                    break;
#endif

#if _POSIX_C_SOURCE >= 200112L
                /* If we're safely using O_TMPFILE, try posix_fallocate(). */
                if(safe && posix_fallocate(fd, outstart, outdend-outstart) == 0)
                    break;
#endif

                /**
                 * DANGER.
                 *
                 * We have failed to resize to file to the minimum required.
                 * Now risking failure from parallel processes when they
                 * attempt parallel writing, unless this is a single process
                 * handling the entire file sequentially from beginning to end.
                 *
                 * Falling out and crossing fingers.
                 */
        }
    }


    /* With file now at full length, advise it will be used sequentially. */
    posix_fadvise(fd, 0, outdend, POSIX_FADV_SEQUENTIAL);
    posix_fadvise(fd, 0, outdend, POSIX_FADV_NOREUSE);


    /**
     * If we are executing safely, try linking the file into place, fully formed.
     * If this fails because the target already exists in place, that means we
     * lost the race, and the file was created by someone else. We only try
     * linkat() if O_TMPFILE is supported; Therefore, if it returns EEXIST,
     * there should be no race condition involved in reopening the file as
     * created by someone else if we retry (barring actively hostile file
     * manipulations).
     *
     * If we are executing unsafely, the file is already in place, already visible,
     * and competing processes may have already seen it in incomplete form.
     * Without kernel support we cannot do much about this.
     */

    if(safe && linkat(AT_FDCWD, procpath, dirfd, f, AT_SYMLINK_FOLLOW) < 0){
        rc = errno;
        close(fd);
        fd = -1;
        if(rc == EEXIST) goto retry_openat_existing;
        else             goto fatal_fits_not_created;
    }
    close(internaldfd);


    /**
     * With an empty FITS file materialized, open it as a pre-existing disk file.
     */

    if(fits_open_diskfile(fptr, procpath, READWRITE, status) ||
       fits_movabs_hdu   (*fptr, 1, NULL,            status)){
        rc = EINVAL;
        goto fatal_fd_close;
    }
    return fd;


    /* Failure Modes */
    if(0){fatal_fits_not_opened:  *status = FILE_NOT_OPENED;
    }else{fatal_fits_not_created: *status = FILE_NOT_CREATED;
    }
    fatal_fd_close:
    close(internaldfd);
    close(fd);
    return -rc;
}
