#include <cuda_runtime.h>
#if !defined(__cplusplus)
#error "__cplusplus is NOT defined"
#elif __cplusplus < 201103L
#error "C++ standard is below C++11: " 
#elif __cplusplus < 201703L
#error "C++ standard is below C++17 (but >= C++11)"
#else
#error "C++ standard is C++17 or above"
#endif

enum class NeuronType : unsigned char {
    EXCITATORY = 0,
};

int main() { return 0; }
