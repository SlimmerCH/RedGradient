#pragma once

#define HIP_CHECK(call) \
do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        std::cerr << "HIP Error: " << hipGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        std::abort(); \
    } \
} while(0)