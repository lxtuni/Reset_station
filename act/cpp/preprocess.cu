// CUDA 前处理：把相机原图变成网络要的输入。全部在显卡上完成，结果直接留在显卡内存里交给 TensorRT。
//
// 输入  src：uint8，按 高×宽×通道 (HWC) 排列，一个像素的 R、G、B 三个数挨在一起
// 输出  dst：float，按 通道×高×宽 (CHW) 排列，先放整张 R，再放整张 G，再放整张 B
// 每个数做的事：  y = (x / 255 - 均值[c]) / 标准差[c]
#include <cuda_runtime.h>
#include <cstdint>

// __global__ 表示这是一个 kernel：由 CPU 发起，在显卡上执行的函数。
// 它会被成千上万个线程同时执行，每个线程负责一个像素。线程靠下面的下标知道自己负责哪个像素。
__global__ void preprocess_kernel(const uint8_t* src, float* dst, int H, int W,
                                  float m0, float m1, float m2, float s0, float s1, float s2) {
    // 线程按“块”组织，块再排成“网格”。这里块和网格都是二维的，正好对应图像的 x、y。
    int x = blockIdx.x * blockDim.x + threadIdx.x;      // 我是第几列
    int y = blockIdx.y * blockDim.y + threadIdx.y;      // 我是第几行
    if (x >= W || y >= H) return;                        // 网格比图像略大（向上取整），多出来的线程什么都不做

    int pix = y * W + x;                                 // 第几个像素
    const uint8_t* p = src + pix * 3;                    // HWC：这个像素的 R、G、B 的位置
    int plane = H * W;                                   // CHW：一个通道占多少个数
    dst[pix]             = (p[0] * (1.0f / 255.0f) - m0) / s0;   // R 平面
    dst[pix + plane]     = (p[1] * (1.0f / 255.0f) - m1) / s1;   // G 平面
    dst[pix + 2 * plane] = (p[2] * (1.0f / 255.0f) - m2) / s2;   // B 平面
}

// 给 C++ 主程序调用的普通函数。d_ 开头的指针都指向显卡内存。
// stream 是一条“任务队列”：放进同一条 stream 的拷贝、kernel、推理会按顺序执行，CPU 不用等它们做完。
void preprocess_gpu(const uint8_t* d_src, float* d_dst, int H, int W,
                    const float* mean, const float* stdv, cudaStream_t stream) {
    dim3 block(16, 16);                                  // 每块 16×16 = 256 个线程
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);   // 需要多少块才能盖住整张图
    preprocess_kernel<<<grid, block, 0, stream>>>(d_src, d_dst, H, W,
                                                  mean[0], mean[1], mean[2], stdv[0], stdv[1], stdv[2]);
}
