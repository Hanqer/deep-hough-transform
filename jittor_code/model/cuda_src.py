cuda_src_forward = '''
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
                                        r = ::round(float(xx) * (tabCos[angleIndex]) + float(yy) * (tabSin[angleIndex]));
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


        using namespace std;

        int blockSizeX = std::min(8, in0_shape3);
        const int threadW = ceil(in0_shape3/(float)blockSizeX);

        int blockSizeY = std::min(8, in0_shape2);
        const int threadH = ceil(in0_shape3/(float)blockSizeY);

        int blockSizeZ = std::min(8, #numangle);
        const int threadK = ceil(#numangle/(float)blockSizeZ);

        const dim3 blocks(in0_shape1, in0_shape0);
        const dim3 threads(blockSizeX, blockSizeY, blockSizeZ);

        cudaMemsetAsync(out0_p, 0, out0->size, 0);
        line_accum_forward_kernel<<<blocks, threads>>>(
            in0_p,
            in1_p,
            in2_p,
            out0_p,
            in0_shape3,
            in0_shape2,
            threadW,
            threadH,
            threadK,
            in0_shape1,
            in0_shape0,
            #numangle,
            #numrho
        );
'''

cuda_src_backward = '''
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

            if (x < imWidth && y < imHeight && channel < channelSize && batch < batchSize && k < numangle)
            {
                int imgIndex = imgStartIdx;
                int angleIndex;
                int outIndex;
                int r;
                for (int idY=0; idY < threadH; idY++)
                {
                    imgIndex = imgStartIdx + idY*imWidth;
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
                                        outIndex = batch*channelSize*numangle*numrho + numangle*numrho*channel + angleIndex*numrho + r;
                                        float val = grad_out[outIndex];    
                                        atomicAdd(&(grad_in[imgIndex]), val);
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


        using namespace std;


        int blockSizeX = std::min(8, in1_shape3);
        const int threadW = ceil(in1_shape3/(float)blockSizeX);

        int blockSizeY = std::min(8, in1_shape2);
        const int threadH = ceil(in1_shape3/(float)blockSizeY);

        int blockSizeZ = std::min(8, #numangle);
        const int threadK = ceil(#numangle/(float)blockSizeZ);

        const dim3 blocks(in1_shape1, in1_shape0);
        const dim3 threads(blockSizeX, blockSizeY, blockSizeZ);

        cudaMemsetAsync(out1_p, 0, out0->size, 0);
        line_accum_backward_kernel<<<blocks, threads>>>(
            out0_p,
            in1_p,
            in2_p,
            in3_p,
            in1_shape3,
            in1_shape2,
            threadW,
            threadH,
            threadK,
            in1_shape1,
            in1_shape0,
            #numangle,
            #numrho
        );
'''
