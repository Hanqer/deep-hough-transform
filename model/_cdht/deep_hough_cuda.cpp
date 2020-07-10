#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>
#include <stdio.h>
#include <math.h>
#include <iostream>

// CUDA forward declarations
std::vector<torch::Tensor> line_accum_cuda_forward(
    const torch::Tensor feat, 
    const float* tabCos,
    const float* tabSin,
    torch::Tensor output,
    const int numangle,
    const int numrho);

std::vector<torch::Tensor> line_accum_cuda_backward(
    torch::Tensor grad_outputs,
    torch::Tensor grad_in,
    torch::Tensor feat,
    const float* tabCos,
    const float* tabSin,
    const int numangle,
    const int numrho);

// C++ interface

#define CHECK_CUDA(x) AT_ASSERT(x.type().is_cuda()) //, #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERT(x.is_contiguous()) //, #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define PI 3.14159265358979323846

void initTab(float* tabSin, float* tabCos, const int numangle, const int numrho, const int H, const int W)
{
    float irho = int(std::sqrt(H*H + W*W) + 1) / float((numrho - 1));
    float itheta = PI / numangle;
    float angle = 0;
    for(int i = 0; i < numangle; ++i)
    {
        tabCos[i] = std::cos(angle) / irho;
        tabSin[i] = std::sin(angle) / irho;
        angle += itheta;
    }
}

std::vector<at::Tensor> line_accum_forward(
    const at::Tensor feat,
    at::Tensor output,
    const int numangle,
    const int numrho) {

    CHECK_INPUT(feat);
    CHECK_INPUT(output);
    float tabSin[numangle], tabCos[numangle];
    const int H = feat.size(2);
    const int W = feat.size(3);
    initTab(tabSin, tabCos, numangle, numrho, H, W);
    const int batch_size = feat.size(0);
    const int channels_size = feat.size(1);
    
    // torch::set_requires_grad(output, true);
    auto out = line_accum_cuda_forward(feat, tabCos, tabSin, output, numangle, numrho);
    // std::cout << out[0].sum() << std::endl;
    CHECK_CONTIGUOUS(out[0]);
    return out;
}

std::vector<torch::Tensor> line_accum_backward(
    torch::Tensor grad_outputs,
    torch::Tensor grad_inputs,
    torch::Tensor feat,
    const int numangle,
    const int numrho) {
  
    CHECK_INPUT(grad_outputs);
    CHECK_INPUT(grad_inputs);
    CHECK_INPUT(feat);

    float tabSin[numangle], tabCos[numangle];
    const int H = feat.size(2);
    const int W = feat.size(3);
    initTab(tabSin, tabCos, numangle, numrho, H, W);

    const int batch_size = feat.size(0);
    const int channels_size = feat.size(1);
    const int imH = feat.size(2);
    const int imW = feat.size(3);

    return line_accum_cuda_backward(
        grad_outputs,
        grad_inputs,
        feat,
        tabCos,
        tabSin,
        numangle,
        numrho);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &line_accum_forward, "line features accumulating forward (CUDA)");
    m.def("backward", &line_accum_backward, "line features accumulating backward (CUDA)");
}