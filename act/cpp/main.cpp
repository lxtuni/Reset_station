// 复位台 ACT 的 C++ 推理程序：相机原图 → CUDA 前处理 → TensorRT 推理 → 还原成真实动作。
// 不依赖 Python 和 PyTorch。用法：./act_infer <引擎文件> <样本文件夹>
//
// 一个控制周期里做的事（run_once）：
//   1. 两张原图 (uint8) 从 CPU 内存拷到显卡
//   2. CUDA kernel 在显卡上做归一化和排列转换（preprocess.cu）
//   3. 6 个关节角度在 CPU 上归一化后拷到显卡
//   4. TensorRT 推理，输出 20×5 的归一化动作
//   5. 输出拷回 CPU，还原成真实动作
// 1–4 步都排进同一条 CUDA stream，CPU 只在最后等一次。
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

void preprocess_gpu(const uint8_t* d_src, float* d_dst, int H, int W,
                    const float* mean, const float* stdv, cudaStream_t stream);   // 在 preprocess.cu 里

static const int H = 240, W = 320, NQ = 6, CHUNK = 20, NA = 5;
static const size_t IMG_U8 = H * W * 3, IMG_F = 3 * H * W;

// TensorRT 要求提供一个日志接收器。这里只打印警告和错误。
class Logger : public nvinfer1::ILogger {
    void log(Severity s, const char* msg) noexcept override {
        if (s <= Severity::kWARNING) std::printf("[TensorRT] %s\n", msg);
    }
};

template <class T> std::vector<T> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { std::printf("打不开 %s\n", path.c_str()); std::exit(1); }
    size_t n = f.tellg(); f.seekg(0);
    std::vector<T> v(n / sizeof(T)); f.read(reinterpret_cast<char*>(v.data()), n);
    return v;
}

// 每个 CUDA 调用都会返回错误码，出错就停下并报行号
#define CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    std::printf("CUDA 错误 %s，第 %d 行\n", cudaGetErrorString(e), __LINE__); std::exit(1); } } while (0)

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}
static void stats(const char* name, std::vector<double> v) {
    std::sort(v.begin(), v.end());
    double mean = 0; for (double x : v) mean += x; mean /= v.size();
    std::printf("%-34s mean=%.2f ms  p50=%.2f  p95=%.2f\n", name, mean, v[v.size() / 2], v[size_t(v.size() * 0.95)]);
}

int main(int argc, char** argv) {
    if (argc < 3) { std::printf("用法：%s <引擎文件> <样本文件夹>\n", argv[0]); return 1; }
    std::string dir = argv[2];

    // ---------- 读入：引擎、一帧原始输入、归一化常数、Python 的参考答案 ----------
    auto blob = read_file<char>(argv[1]);
    auto top = read_file<uint8_t>(dir + "/top.u8"), wrist = read_file<uint8_t>(dir + "/wrist.u8");
    auto state = read_file<float>(dir + "/state.f32"), norm = read_file<float>(dir + "/norm.f32");
    auto ref_top = read_file<float>(dir + "/ref_top.f32"), ref_act = read_file<float>(dir + "/ref_actions.f32");
    // norm.f32 里 34 个数的顺序见 dump_raw_sample.py
    const float *m_top = &norm[0], *s_top = &norm[3], *m_wr = &norm[6], *s_wr = &norm[9];
    const float *m_q = &norm[12], *s_q = &norm[18], *m_a = &norm[24], *s_a = &norm[29];

    // ---------- 加载 TensorRT 引擎 ----------
    // 引擎文件是 TensorRT 针对这块显卡编译好的网络。runtime 把它读进来，context 是一次推理要用的工作区。
    Logger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
    std::unique_ptr<nvinfer1::ICudaEngine> engine(runtime->deserializeCudaEngine(blob.data(), blob.size()));
    if (!engine) { std::printf("引擎加载失败\n"); return 1; }
    std::unique_ptr<nvinfer1::IExecutionContext> ctx(engine->createExecutionContext());
    std::printf("引擎的输入输出：");
    for (int i = 0; i < engine->getNbIOTensors(); i++) std::printf("%s ", engine->getIOTensorName(i));
    std::printf("\n");

    // ---------- 分配内存 ----------
    // h_ 开头在 CPU 上，d_ 开头在显卡上。
    // CPU 这边用“锁页内存”（cudaMallocHost）：操作系统保证它不被换出，拷到显卡更快，也才能做异步拷贝。
    cudaStream_t stream; CHECK(cudaStreamCreate(&stream));
    uint8_t *h_top, *h_wrist; float *h_q, *h_act;
    CHECK(cudaMallocHost(&h_top, IMG_U8)); CHECK(cudaMallocHost(&h_wrist, IMG_U8));
    CHECK(cudaMallocHost(&h_q, NQ * sizeof(float))); CHECK(cudaMallocHost(&h_act, CHUNK * NA * sizeof(float)));
    uint8_t *d_top_u8, *d_wrist_u8; float *d_top, *d_wrist, *d_q, *d_act;
    CHECK(cudaMalloc(&d_top_u8, IMG_U8)); CHECK(cudaMalloc(&d_wrist_u8, IMG_U8));
    CHECK(cudaMalloc(&d_top, IMG_F * sizeof(float))); CHECK(cudaMalloc(&d_wrist, IMG_F * sizeof(float)));
    CHECK(cudaMalloc(&d_q, NQ * sizeof(float))); CHECK(cudaMalloc(&d_act, CHUNK * NA * sizeof(float)));
    // 告诉 TensorRT 每个输入输出在显卡内存的哪里。名字是导出 ONNX 时起的。
    ctx->setTensorAddress("top", d_top); ctx->setTensorAddress("wrist", d_wrist);
    ctx->setTensorAddress("state", d_q); ctx->setTensorAddress("actions", d_act);

    std::vector<float> actions(CHUNK * NA);
    // sync_each=true 时每一段后面都等显卡做完，用来分段计时；正常运行时只在最后等一次
    auto run_once = [&](bool sync_each, double* t) {
        double t0 = now_ms();
        std::memcpy(h_top, top.data(), IMG_U8); std::memcpy(h_wrist, wrist.data(), IMG_U8);   // 模拟“相机给了新的一帧”
        for (int i = 0; i < NQ; i++) h_q[i] = (state[i] - m_q[i]) / s_q[i];                    // 关节角度只有 6 个数，在 CPU 上算
        CHECK(cudaMemcpyAsync(d_top_u8, h_top, IMG_U8, cudaMemcpyHostToDevice, stream));
        CHECK(cudaMemcpyAsync(d_wrist_u8, h_wrist, IMG_U8, cudaMemcpyHostToDevice, stream));
        CHECK(cudaMemcpyAsync(d_q, h_q, NQ * sizeof(float), cudaMemcpyHostToDevice, stream));
        preprocess_gpu(d_top_u8, d_top, H, W, m_top, s_top, stream);
        preprocess_gpu(d_wrist_u8, d_wrist, H, W, m_wr, s_wr, stream);
        if (sync_each) CHECK(cudaStreamSynchronize(stream));
        double t1 = now_ms();
        if (!ctx->enqueueV3(stream)) { std::printf("推理失败\n"); std::exit(1); }               // 把推理排进同一条 stream
        if (sync_each) CHECK(cudaStreamSynchronize(stream));
        double t2 = now_ms();
        CHECK(cudaMemcpyAsync(h_act, d_act, CHUNK * NA * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CHECK(cudaStreamSynchronize(stream));                                                  // 等显卡把以上所有事做完
        for (int i = 0; i < CHUNK * NA; i++) actions[i] = h_act[i] * s_a[i % NA] + m_a[i % NA]; // 还原成真实动作
        double t3 = now_ms();
        if (t) { t[0] = t1 - t0; t[1] = t2 - t1; t[2] = t3 - t2; t[3] = t3 - t0; }
    };

    // ---------- 核对 1：CUDA 前处理和 Python 前处理是否一致 ----------
    run_once(false, nullptr);
    std::vector<float> got(IMG_F);
    CHECK(cudaMemcpy(got.data(), d_top, IMG_F * sizeof(float), cudaMemcpyDeviceToHost));
    float e_pre = 0; for (size_t i = 0; i < IMG_F; i++) e_pre = std::max(e_pre, std::fabs(got[i] - ref_top[i]));
    // ---------- 核对 2：整条链路的最终动作和 Python (PyTorch) 是否一致 ----------
    float e_act = 0; for (int i = 0; i < CHUNK * NA; i++) e_act = std::max(e_act, std::fabs(actions[i] - ref_act[i]));
    std::printf("CUDA 前处理和 Python 的最大误差：%.2e\n", e_pre);
    std::printf("最终动作和 Python 的最大误差：  %.2e\n", e_act);
    std::printf("第 1 步动作：C++  [%.4f %.4f %.4f %.4f %.4f]\n", actions[0], actions[1], actions[2], actions[3], actions[4]);
    std::printf("            Python [%.4f %.4f %.4f %.4f %.4f]\n", ref_act[0], ref_act[1], ref_act[2], ref_act[3], ref_act[4]);

    // ---------- 计时 ----------
    for (int i = 0; i < 50; i++) run_once(false, nullptr);                                     // 预热
    std::vector<double> a, b, c, d, e; double t[4];
    for (int i = 0; i < 300; i++) { run_once(true, t); a.push_back(t[0]); b.push_back(t[1]); c.push_back(t[2]); }
    for (int i = 0; i < 300; i++) { run_once(false, t); d.push_back(t[3]); }
    // 对照：同样的前处理在 CPU 上单线程做两张图要多久
    std::vector<float> cpu(IMG_F);
    for (int k = 0; k < 300; k++) {
        double t0 = now_ms();
        for (const auto* img : {&top, &wrist})
            for (int p = 0; p < H * W; p++)
                for (int ch = 0; ch < 3; ch++)
                    cpu[ch * H * W + p] = ((*img)[p * 3 + ch] * (1.0f / 255.0f) - m_top[ch]) / s_top[ch];
        e.push_back(now_ms() - t0);
    }
    std::printf("\n分段计时（每段后都等显卡做完）：\n");
    stats("  上传两张图 + CUDA 前处理", a); stats("  TensorRT 推理", b); stats("  取回结果 + 还原动作", c);
    std::printf("正常运行（只在最后等一次）：\n"); stats("  原图 → 动作，整条链路", d);
    std::printf("对照：\n"); stats("  同样的前处理在 CPU 上（单线程）", e);

    cudaFree(d_top_u8); cudaFree(d_wrist_u8); cudaFree(d_top); cudaFree(d_wrist); cudaFree(d_q); cudaFree(d_act);
    cudaFreeHost(h_top); cudaFreeHost(h_wrist); cudaFreeHost(h_q); cudaFreeHost(h_act);
    cudaStreamDestroy(stream);
    return 0;
}
