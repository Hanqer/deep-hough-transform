#include <ATen/ATen.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm> 
#include <math.h> 
#include <stdio.h>
#include <iostream>


// -------
// KERNELS 
// ------- 
__global__ void helloCUDA(const float *f)
{
    for(int i = 0; i < 10; ++i)
    {
        printf("%d ", f[i]);
    }
    printf("\n");
    // printf("Hello thread %d, %d, %d, f=%f\n", threadIdx.x, threadIdx.y, threadIdx.z, f);
}


__global__
void line_accum_forward_kernel(
    const float* __restrict__ feat,
    const float*  tabCos,
    const float*  tabSin,
    float* output,
    const int imWidth,
    const int imHeight, 
    const int threadW, 
    const int threadH,
    const int threadK,
    const int channelSize, 
    const int batchSize,
    const int numangle,
    const int numrho)
{
    int batch = blockIdx.y;
    int channel = blockIdx.x;
    int x = threadIdx.x*threadW; 
    int y = threadIdx.y*threadH;
    int k = threadIdx.z*threadK;

    int imgStartIdx = batch*channelSize*imWidth*imHeight+
                      channel*imWidth*imHeight+
                      y*imWidth+
                      x;
    
    int angleStartIdx = k;
    
    if (x < imWidth && y < imHeight && channel < channelSize && batch < batchSize && k < numangle)
    {
        int imgIndex = imgStartIdx;
        int angleIndex;
        int outIndex;
        int r;
        for (int idY=0; idY < threadH; idY++)
        {
            imgIndex = imgStartIdx + idY*imWidth;
            // labelIndex = labelStartIdx + idY*imWidth;
            if (y+idY < imHeight)
            {
                for (int idX=0; idX<threadW; idX++)
                {
                    if (x + idX < imWidth)
                    {
                        for (int idK=0; idK<threadK; idK++)
                        {
                            angleIndex = angleStartIdx + idK;
                            if(angleIndex < numangle)
                            {
                                int xx = x + idX - imWidth / 2, yy = y + idY - imHeight / 2;
                                r = std::round(float(xx) * (tabCos[angleIndex]) + float(yy) * (tabSin[angleIndex]));
                                r += ((numrho) / 2);
                                outIndex = batch*channelSize*numangle*numrho + numangle*numrho*channel + angleIndex*numrho + r;
                                float val = feat[imgIndex];
                                atomicAdd(&(output[outIndex]), val);
                                
                            }
                            else break;
                        }
                        imgIndex++;
                    }
                    else break;
                }
            }
            else break;
        }
    }
}


__global__
void line_accum_backward_kernel(
    float* grad_in,
    const float* grad_out,
    const float* tabCos,
    const float* tabSin,
    const int imWidth,
    const int imHeight, 
    const int threadW, 
    const int threadH,
    const int threadK,
    const int channelSize, 
    const int batchSize,
    const int numangle,
    const int numrho)
{
    int batch = blockIdx.y;
    int channel = blockIdx.x;
    int x = threadIdx.x*threadW; 
    int y = threadIdx.y*threadH;
    int k = threadIdx.z*threadK;

    int imgStartIdx = batch*channelSize*imWidth*imHeight+
                      channel*imWidth*imHeight+
                      y*imWidth+
                      x;
    
    int angleStartIdx = k;
    // int maxindex = batchSize * channelSize * imHeight * imWidth;
    if (x < imWidth && y < imHeight && channel < channelSize && batch < batchSize && k < numangle)
    {
        int imgIndex = imgStartIdx;
        int angleIndex;
        int outIndex;
        int r;
        for (int idY=0; idY < threadH; idY++)
        {
            imgIndex = imgStartIdx + idY*imWidth;
            // labelIndex = labelStartIdx + idY*imWidth;
            if (y+idY < imHeight)
            {
                for (int idX=0; idX<threadW; idX++)
                {
                    if (x + idX < imWidth)
                    {
                        for (int idK=0; idK<threadK; idK++)
                        {
                            angleIndex = angleStartIdx + idK;
                            if(angleIndex < numangle)
                            {
                                int xx = x + idX - imWidth / 2, yy = y + idY - imHeight / 2;
                                r = std::round(float(xx)*tabCos[angleIndex] + float(yy)*tabSin[angleIndex]);
                                r += ((numrho) / 2);
                                // printf("ppp=%p\n", grad_out);
                                // output shape : [N, C, numangle, numrho]
                                outIndex = batch*channelSize*numangle*numrho + numangle*numrho*channel + angleIndex*numrho + r;
                                // printf("Index: %d\n", outIndex);
                                float val = grad_out[outIndex];
                                // float val = float(1e-7);
                                // printf("ok, this line running. val=%.4f", val);
                                atomicAdd(&(grad_in[imgIndex]), val);
                                // atomicAdd(&(grad_in[imgIndex]), val);
                                // printf("Val = %.15f, grad_in_add = %.15f, imgIndex = %d\n", val, grad_in[imgIndex], imgIndex);
                            }
                            else break;
                        }
                        imgIndex++;
                    }
                    else break;
                }
            }
            else break;
        }
    }
    // printf("grad_in_add = %.15f, imgIndex = %d\n", grad_in[52134], 52134);
}


// ---------
// Wrappers
// --------- 

std::vector<torch::Tensor> line_accum_cuda_forward(
    const torch::Tensor feat,
    const float* tabCos,
    const float* tabSin, 
    torch::Tensor output,
    const int numangle,
    const int numrho){
    // -feat: [N, C, H, W]
    // -tabCos: [numangle]
    // -tabSin: [numangle]
    const int batch_size = feat.size(0);
    const int channels_size = feat.size(1);
    const int imH = feat.size(2); 
    const int imW = feat.size(3);

    int blockSizeX = std::min(8, imW);
    const int threadW    = ceil(imW/(float)blockSizeX);

    int blockSizeY = std::min(8, imH);
    const int threadH    = ceil(imH/(float)blockSizeY);

    int blockSizeZ = std::min(8, numangle);
    const int threadK = ceil(numangle/(float)blockSizeZ);

    const dim3 blocks(channels_size, batch_size);
    const dim3 threads(blockSizeX, blockSizeY, blockSizeZ);

    float *d_tabCos, *d_tabSin;

    cudaMalloc((void **)&d_tabCos, sizeof(float)*numangle);
    cudaMalloc((void **)&d_tabSin, sizeof(float)*numangle);

    cudaMemcpy(d_tabCos, tabCos, sizeof(float)*numangle, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tabSin, tabSin, sizeof(float)*numangle, cudaMemcpyHostToDevice);

    // std::cout << imW << " " << imH << " " << channels_size << " " << batch_size << " " << numangle << " " << numrho << std::endl;
    line_accum_forward_kernel<<<blocks, threads>>>(
        feat.data<float>(),
        d_tabCos,
        d_tabSin,
        output.data<float>(),
        imW,
        imH,
        threadW,
        threadH,
        threadK,
        channels_size,
        batch_size,
        numangle,
        numrho
    );
    // helloCUDA<<<blocks, threads>>>(tabCos);
    // cudaDeviceSynchronize();
    // std::cout << output << std::endl;
    // std::cout << output.sum() << std::endl;
    cudaFree(d_tabCos);
    cudaFree(d_tabSin);
    return {output};
}

std::vector<torch::Tensor> line_accum_cuda_backward(
    torch::Tensor grad_outputs,
    torch::Tensor grad_in,
    torch::Tensor feat,
    const float* tabCos,
    const float* tabSin,
    const int numangle,
    const int numrho)
{
    const int batch_size = feat.size(0);
    const int channels_size = feat.size(1);
    const int imH = feat.size(2); 
    const int imW = feat.size(3);

    int blockSizeX = std::min(8, imW);
    const int threadW    = ceil(imW/(float)blockSizeX);

    int blockSizeY = std::min(8, imH);
    const int threadH    = ceil(imH/(float)blockSizeY);

    int blockSizeZ = std::min(8, numangle);
    const int threadK = ceil(numangle/(float)blockSizeZ);

    const dim3 blocks(channels_size, batch_size);
    const dim3 threads(blockSizeX, blockSizeY, blockSizeZ);

    float *d_tabCos, *d_tabSin;

    cudaMalloc((void **)&d_tabCos, sizeof(float)*numangle);
    cudaMalloc((void **)&d_tabSin, sizeof(float)*numangle);

    cudaMemcpy(d_tabCos, tabCos, sizeof(float)*numangle, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tabSin, tabSin, sizeof(float)*numangle, cudaMemcpyHostToDevice);
    // std::cout << imW << " " << imH << " " << channels_size << " " << batch_size << " " << numangle << " " << numrho << std::endl;
    
    
    // printf("p = %p\n", grad_outputs.data<float>());
    // printf("p = %p\n", grad_in.data<float>());
    
    line_accum_backward_kernel<<<blocks, threads>>>(
        grad_in.data<float>(),
        grad_outputs.data<float>(),
        d_tabCos,
        d_tabSin,
        imW,
        imH,
        threadW,
        threadH,
        threadK,
        channels_size,
        batch_size,
        numangle,
        numrho
    );
    // printf("p = %p\n", grad_outputs.data<float>());
    // printf("p = %p\n", grad_in.data<float>());
    // std::cout << grad_outputs << std::endl;
    // cudaDeviceSynchronize();
    cudaFree(d_tabCos);
    cudaFree(d_tabSin);
    return {grad_in};
}
