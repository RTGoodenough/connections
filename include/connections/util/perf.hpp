#pragma once

#ifdef CNTNS_PERF_TESTING

#define CNTNS_PERF_FUNC __attribute__((noinline))

#define CNTNS_INLINE

#else

#define CNTNS_PERF_FUNC

#define CNTNS_INLINE __attribute__((always_inline))

#endif