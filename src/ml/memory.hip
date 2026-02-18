#include <hip/hip_runtime.h>
#include "ml/kernel/utils.hpp"
#include <stdexcept>
#include <vector>
#include <utility>

template <typename T>
class GPUArray {
private:
    T* data_ = nullptr;
    size_t size_ = 0; // number of elements

public:
    // constructor: allocate memory
    explicit GPUArray(size_t size) : size_(size) {
        HIP_CHECK(hipMalloc(&data_, size * sizeof(T)));
    }

    // destructor frees memory automatically
    ~GPUArray() {
        if (data_) {
            hipFreeAsync(data_);
        }
    }

    // disable copy contructor and copy assignment (prevent double-free memory)
    GPUArray(const GPUArray&) = delete;
    GPUArray& operator=(const GPUArray&) = delete;

    // move constructor
    GPUArray(GPUArray&& other) noexcept 
        : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    // move assignment
    GPUArray& operator=(GPUArray&& other) noexcept {
        if (this != &other) {
            if (data_) hipFree(data_); // clean up current
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    // copy to device (Host -> GPU)
    void to_device(T* host_ptr) {
        HIP_CHECK(hipMemcpy(data_, host_ptr, size_ * sizeof(T), hipMemcpyHostToDevice));
    }

    // copy from device (GPU -> Host)
    void to_host(T* host_ptr) const {
        HIP_CHECK(hipMemcpy(host_ptr, data_, size_ * sizeof(T), hipMemcpyDeviceToHost));
    }

    // accessor for kernels
    T* data() { return data_; }
    const T* data() const { return data_; }
    size_t size() const { return size_; }
};