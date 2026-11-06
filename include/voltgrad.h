/**
 * VoltGrad C-Compatible Application Binary Interface (ABI)
 *
 * Formalized under ADR-0004: Language-Independent Engine Core with Stable
 * C-Compatible ABI & Pluggable Backends (Go-First, Not Go-Only).
 *
 * This header serves as the canonical contract between the VoltGrad compute engine
 * and language bindings (Go via purego, Rust, Python, C/C++, Zig).
 */

#ifndef VOLTGRAD_H
#define VOLTGRAD_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* --- Platform Export Definitions --- */
#if defined(_WIN32) || defined(__CYGWIN__)
  #if defined(VOLTGRAD_BUILD_LIBRARY)
    #define VOLTGRAD_API __declspec(dllexport)
  #else
    #define VOLTGRAD_API __declspec(dllimport)
  #endif
#else
  #if defined(__GNUC__) && __GNUC__ >= 4
    #define VOLTGRAD_API __attribute__((visibility("default")))
  #else
    #define VOLTGRAD_API
  #endif
#endif

/* --- Version Information --- */
#define VOLTGRAD_VERSION_MAJOR 0
#define VOLTGRAD_VERSION_MINOR 4
#define VOLTGRAD_VERSION_PATCH 0

/* --- Status Codes --- */
typedef enum {
    VOLTGRAD_SUCCESS                   = 0,
    VOLTGRAD_ERROR_INVALID_ARGUMENT    = -1,
    VOLTGRAD_ERROR_DEVICE_NOT_FOUND    = -2,
    VOLTGRAD_ERROR_OUT_OF_MEMORY       = -3,
    VOLTGRAD_ERROR_DEVICE_LOST         = -4,
    VOLTGRAD_ERROR_TIMEOUT             = -5,
    VOLTGRAD_ERROR_PRECISION_OVERFLOW  = -6,
    VOLTGRAD_ERROR_BACKEND_UNAVAILABLE = -7,
    VOLTGRAD_ERROR_INTERNAL            = -99
} VoltGradStatus;

/* --- Backend Types --- */
typedef enum {
    VOLTGRAD_BACKEND_AUTO   = 0,
    VOLTGRAD_BACKEND_VULKAN = 1,
    VOLTGRAD_BACKEND_CPU    = 2
} VoltGradBackendType;

/* --- Memory Residency --- */
typedef enum {
    VOLTGRAD_RESIDENCY_HOST              = 0,
    VOLTGRAD_RESIDENCY_DEVICE_PERSISTENT = 1,
    VOLTGRAD_RESIDENCY_SHARED            = 2
} VoltGradResidency;

/* --- Optimizer Types --- */
typedef enum {
    VOLTGRAD_OPTIMIZER_SGD      = 0,
    VOLTGRAD_OPTIMIZER_MOMENTUM = 1,
    VOLTGRAD_OPTIMIZER_ADAMQ    = 2
} VoltGradOptimizerType;

/* --- Opaque Handles --- */
typedef struct VoltGradEngine*   VoltGradEngineHandle;
typedef struct VoltGradTensor*   VoltGradTensorHandle;
typedef struct VoltGradGraph*    VoltGradGraphHandle;
typedef struct VoltGradToken*    VoltGradTokenHandle;

/* --- Engine Configuration --- */
typedef struct {
    VoltGradBackendType backend_type;
    uint32_t            device_index;
    int32_t             force_32bit;       /* Boolean: 1 forces 32-bit execution (ADR-0002) */
    int32_t             enable_profiling;  /* Boolean: 1 enables timestamp queries */
} VoltGradEngineConfig;

/* --- Device Report --- */
typedef struct {
    char        device_name[256];
    char        driver_info[256];
    uint32_t    vendor_id;
    uint32_t    device_id;
    uint64_t    total_memory_bytes;
    int32_t     supports_shader_int64; /* Boolean */
    int32_t     is_64bit_execution;    /* Boolean */
    int32_t     max_gpu_bit_width;     /* 32 or 64 */
} VoltGradDeviceReport;

/* --- Tensor Descriptor --- */
typedef struct {
    uint32_t           rank;
    const int64_t*     shape;             /* Array of dimension lengths */
    int32_t            bit_width;         /* Tensor-wise DBW bit-width: [2, 64] */
    int32_t            fixed_point_scale; /* Fractional bits, e.g. 16 for Q16 */
    VoltGradResidency  residency;          /* Host or GPU resident */
} VoltGradTensorDesc;

/* --- Memory Slice (Contiguous View) --- */
typedef struct {
    void*  ptr;
    size_t byte_len;
} VoltGradSlice;

/* ========================================================================= */
/* --- Core Engine Lifecycle --- */
/* ========================================================================= */

VOLTGRAD_API VoltGradStatus voltgrad_engine_create(
    const VoltGradEngineConfig* config,
    VoltGradEngineHandle*       out_engine
);

VOLTGRAD_API VoltGradStatus voltgrad_engine_get_report(
    VoltGradEngineHandle  engine,
    VoltGradDeviceReport* out_report
);

VOLTGRAD_API VoltGradStatus voltgrad_engine_destroy(
    VoltGradEngineHandle engine
);

/* ========================================================================= */
/* --- Tensor Lifecycle & Memory --- */
/* ========================================================================= */

VOLTGRAD_API VoltGradStatus voltgrad_tensor_create(
    VoltGradEngineHandle      engine,
    const VoltGradTensorDesc* desc,
    VoltGradTensorHandle*     out_tensor
);

VOLTGRAD_API VoltGradStatus voltgrad_tensor_get_desc(
    VoltGradTensorHandle tensor,
    VoltGradTensorDesc*  out_desc
);

VOLTGRAD_API VoltGradStatus voltgrad_tensor_copy_from_host(
    VoltGradTensorHandle tensor,
    const void*          host_ptr,
    size_t               byte_len
);

VOLTGRAD_API VoltGradStatus voltgrad_tensor_copy_to_host(
    VoltGradTensorHandle tensor,
    void*                host_ptr,
    size_t               byte_len
);

VOLTGRAD_API VoltGradStatus voltgrad_tensor_destroy(
    VoltGradTensorHandle tensor
);

/* ========================================================================= */
/* --- Asynchronous Execution & Graph Capture --- */
/* ========================================================================= */

VOLTGRAD_API VoltGradStatus voltgrad_graph_create(
    VoltGradEngineHandle engine,
    VoltGradGraphHandle* out_graph
);

VOLTGRAD_API VoltGradStatus voltgrad_graph_add_matmul_forward(
    VoltGradGraphHandle  graph,
    VoltGradTensorHandle x,
    VoltGradTensorHandle w,
    VoltGradTensorHandle bias,  /* Nullable */
    VoltGradTensorHandle out
);

VOLTGRAD_API VoltGradStatus voltgrad_graph_add_relu(
    VoltGradGraphHandle  graph,
    VoltGradTensorHandle in_tensor,
    VoltGradTensorHandle out_tensor
);

VOLTGRAD_API VoltGradStatus voltgrad_graph_add_optimizer_step(
    VoltGradGraphHandle   graph,
    VoltGradOptimizerType opt_type,
    VoltGradTensorHandle  param,
    VoltGradTensorHandle  grad,
    VoltGradTensorHandle  velocity,
    int64_t               learning_rate,
    int64_t               momentum_or_beta1,
    int64_t               weight_decay
);

VOLTGRAD_API VoltGradStatus voltgrad_graph_submit(
    VoltGradGraphHandle  graph,
    VoltGradTokenHandle* out_token
);

VOLTGRAD_API VoltGradStatus voltgrad_graph_destroy(
    VoltGradGraphHandle graph
);

/* ========================================================================= */
/* --- Synchronization & Diagnostics --- */
/* ========================================================================= */

VOLTGRAD_API VoltGradStatus voltgrad_token_wait(
    VoltGradTokenHandle token,
    uint64_t            timeout_ns
);

VOLTGRAD_API VoltGradStatus voltgrad_token_get_duration_ns(
    VoltGradTokenHandle token,
    uint64_t*           out_duration_ns
);

VOLTGRAD_API VoltGradStatus voltgrad_token_destroy(
    VoltGradTokenHandle token
);

VOLTGRAD_API const char* voltgrad_last_error_message(void);

#ifdef __cplusplus
}
#endif

#endif /* VOLTGRAD_H */
