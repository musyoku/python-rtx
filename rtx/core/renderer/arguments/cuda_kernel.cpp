#include "cuda_kernel.h"

namespace rtx {
CUDAKernelLaunchArguments::CUDAKernelLaunchArguments()
{
    _num_threads = 256;
    _num_blocks = 1024;
}
int CUDAKernelLaunchArguments::num_threads()
{
    return _num_threads;
}
void CUDAKernelLaunchArguments::set_num_threads(int num)
{
    _num_threads = num;
}

int CUDAKernelLaunchArguments::num_blocks()
{
    return _num_blocks;
}
void CUDAKernelLaunchArguments::set_num_blocks(int bounce)
{
    _num_blocks = bounce;
}
}