//
// httpd.ispc.h
// (Header automatically generated by the ispc compiler.)
// DO NOT EDIT THIS FILE.
//

#ifndef ISPC_HTTPD_ISPC_H
#define ISPC_HTTPD_ISPC_H

#include <stdint.h>



#ifdef __cplusplus
namespace ispc { /* namespace */
#endif // __cplusplus

#ifndef __ISPC_ALIGN__
#if defined(__clang__) || !defined(_MSC_VER)
// Clang, GCC, ICC
#define __ISPC_ALIGN__(s) __attribute__((aligned(s)))
#define __ISPC_ALIGNED_STRUCT__(s) struct __ISPC_ALIGN__(s)
#else
// Visual Studio
#define __ISPC_ALIGN__(s) __declspec(align(s))
#define __ISPC_ALIGNED_STRUCT__(s) __ISPC_ALIGN__(s) struct
#endif
#endif

#ifndef __ISPC_STRUCT_outputBuffer__
#define __ISPC_STRUCT_outputBuffer__
struct outputBuffer {
    int32_t outputBytes[];
};
#endif

#ifndef __ISPC_STRUCT_inputBuffer__
#define __ISPC_STRUCT_inputBuffer__
struct inputBuffer {
    int32_t inputBytes[];
};
#endif

#ifndef __ISPC_STRUCT_heapBuffer__
#define __ISPC_STRUCT_heapBuffer__
struct heapBuffer {
    int32_t heap[];
};
#endif

#ifndef __ISPC_STRUCT_requestBuffer__
#define __ISPC_STRUCT_requestBuffer__
struct requestBuffer {
    int32_t request[];
};
#endif

#ifndef __ISPC_STRUCT_responseBuffer__
#define __ISPC_STRUCT_responseBuffer__
struct responseBuffer {
    int32_t response[];
};
#endif


///////////////////////////////////////////////////////////////////////////
// Functions exported from ispc code
///////////////////////////////////////////////////////////////////////////
#if defined(__cplusplus) && (! defined(__ISPC_NO_EXTERN_C) || !__ISPC_NO_EXTERN_C )
extern "C" {
#endif // __cplusplus
    extern void httpd_get_workgroup_size(int32_t &wg_x, int32_t &wg_y, int32_t &wg_z);
    extern void runner_main(int32_t * work_groups, struct inputBuffer &v_94, struct outputBuffer &v_656, struct heapBuffer &_901);
#if defined(__cplusplus) && (! defined(__ISPC_NO_EXTERN_C) || !__ISPC_NO_EXTERN_C )
} /* end extern C */
#endif // __cplusplus


#ifdef __cplusplus
} /* namespace */
#endif // __cplusplus

#endif // ISPC_HTTPD_ISPC_H
