/*
gemm 朴素实现
weight 为两维
*/

// (128, 512) x (512, 128)===>(128, 128)
// (256,128,512) x (512,128)===>(256,128,128)
// (32,256,128,512) x (512,128)===>(32,256,128,128)
// (1,332,256) X (256,2048) ===>(1,332,2048)
__global__ void matmul_kernel_half_mode0(
                                __half *input,                             
                                __half *output,
                                __half *weight, 
                                __half *bias, 
                                int32_t batch_size, 
                                int32_t in_feature,
                                int32_t out_feature) {
    
    int32_t index_row = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t index_col = blockIdx.y * blockDim.y + threadIdx.y;

    if (index_row >= batch_size || index_col >= out_feature) return;
    
    int32_t offset_input = index_row * in_feature;
    int32_t offset_write = index_row * out_feature + index_col;
    float temp = 0.0;
    for (int i = 0; i < in_feature; i++){
        temp += __half2float(input[offset_input + i]) * __half2float(weight[index_col + out_feature * i]);
    }
    
    if (bias) temp += __half2float(bias[index_col]);  
    
    /*
    if (index_row == 0 && index_col == 0){
        for (int i = 0; i < in_feature; i++){
            float cur_res = __half2float(input[offset_input + i]) * __half2float(weight[index_col + out_feature * i]);
            printf("idx: %d, in: %f, weight: %f, res: %f\n", i, __half2float(input[offset_input + i]), __half2float(weight[index_col + out_feature * i]), cur_res);
        }
        printf("%f, %f, %f\n", __half2float(bias[0]), __half2float(bias[1]), __half2float(bias[2]));
    }
    */

    // if (activation == "gelu"){

    //     float a = sqrtf(2 / PI);
    //     float b = temp + 0.044715 * temp * temp * temp;
    //     float t = a * b;

    //     float ex = _exp(t);
    //     ex = ex > 1.0e10 ? 1.0e10: (ex < -1.0e10 ? -1.0e10: ex);
    //     float ex_ = _exp(-t);
    //     ex_ = ex_ > 1.0e10 ? 1.0e10: (ex_ < -1.0e10 ? -1.0e10: ex_);

    //     float m = ex - ex_;
    //     float n = ex + ex_;
    //     // float tanh_res = (ex - ex_) / (ex + ex_);
    //     float tanh_res = m / n;
    //     float res = 0.5 * (1.0 + tanh_res);
    //     temp *= res;
    //     /*
    //     if (index_row == 11 && index_col == 52){
    //         printf("a: %f\n", a);
    //         printf("b: %f\n", b);
    //         printf("t: %f\n", t);
    //         printf("ex: %f\n", ex);
    //         printf("ex_: %f\n", ex_);
    //         printf("minus: %f\n", ex - ex_);
    //         printf("sum: %f\n", ex + ex_);

    //         printf("tanh: %f\n", tanh_res);
    //         printf("res: %f\n", res);
    //         printf("temp: %f\n", temp);
    //     }
    //     */
    // }else if(activation == "relu") {
    //     temp = temp > 0 ? temp : 0 ;
    // }
    
    temp = temp > 0 ? temp : 0 ;

    __syncthreads();
    output[offset_write] = __float2half(temp);   
}

void print_element(half* data_ptr, int size) {
    std::vector<half> h_data(size);
    CUDA_CHECK(cudaMemcpy(h_data.data(), data_ptr, size * sizeof(half), cudaMemcpyDeviceToHost));
    for (int i = 0; i < size; ++i) {
        std::cout << __half2float(h_data.at(i)) << " ";
    }
    std::cout << std::endl;
}

void matmul_half(__half *input,
                __half *output,
                __half *weight,
                __half *bias, 
                int32_t batch_size,  
                int32_t in_feature,
                int32_t out_feature,
                cudaStream_t stream) {
        
    dim3 dim_block(32, 32, 1);
    dim3 dim_grid(DivUp(batch_size, 32), DivUp(out_feature, 32), 1);
    
    // std::cout << "batch_size: " << batch_size << std::endl;
    // std::cout << "in_feature: " << in_feature << std::endl;
    // std::cout << "out_feature: " << out_feature << std::endl;

    matmul_kernel_half_mode0<<<dim_grid, dim_block, 0, stream>>>(
        input,
        output,
        weight,
        bias, 
        batch_size, 
        in_feature, 
        out_feature
        );
    // print_element(output, batch_size*out_feature); 
}


// weight 转置
// (1,332,256) X (2048，256) ===>(1,332,2048)
__global__ void matmul_kernel_half_mode1(
                                __half *input,                             
                                __half *output,
                                __half *weight, 
                                __half *bias, 
                                int32_t batch_size, 
                                int32_t in_feature,
                                int32_t out_feature) {
    
    int32_t index_row = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t index_col = blockIdx.y * blockDim.y + threadIdx.y;

    if (index_row >= batch_size || index_col >= out_feature) return;
    
    int32_t offset_input = index_row * in_feature;
    int32_t offset_weight = index_col*in_feature;
    int32_t offset_write = index_row * out_feature + index_col;
    float temp = 0.0;
    for (int i = 0; i < in_feature; i++){
        // temp += __half2float(input[offset_input + i]) * __half2float(weight[index_col + out_feature * i]);
        temp += __half2float(input[offset_input + i]) * __half2float(weight[offset_weight + i]);
    }
    
    if (bias) temp += __half2float(bias[index_col]);  // diffmax: 2.349989175796509, sim: [[0.9054079]]
    
    /*
    if (index_row == 0 && index_col == 0){
        for (int i = 0; i < in_feature; i++){
            float cur_res = __half2float(input[offset_input + i]) * __half2float(weight[index_col + out_feature * i]);
            printf("idx: %d, in: %f, weight: %f, res: %f\n", i, __half2float(input[offset_input + i]), __half2float(weight[index_col + out_feature * i]), cur_res);
        }
        printf("%f, %f, %f\n", __half2float(bias[0]), __half2float(bias[1]), __half2float(bias[2]));
    }
    */

    // if (activation == "gelu"){

    //     float a = sqrtf(2 / PI);
    //     float b = temp + 0.044715 * temp * temp * temp;
    //     float t = a * b;

    //     float ex = _exp(t);
    //     ex = ex > 1.0e10 ? 1.0e10: (ex < -1.0e10 ? -1.0e10: ex);
    //     float ex_ = _exp(-t);
    //     ex_ = ex_ > 1.0e10 ? 1.0e10: (ex_ < -1.0e10 ? -1.0e10: ex_);

    //     float m = ex - ex_;
    //     float n = ex + ex_;
    //     // float tanh_res = (ex - ex_) / (ex + ex_);
    //     float tanh_res = m / n;
    //     float res = 0.5 * (1.0 + tanh_res);
    //     temp *= res;
    //     /*
    //     if (index_row == 11 && index_col == 52){
    //         printf("a: %f\n", a);
    //         printf("b: %f\n", b);
    //         printf("t: %f\n", t);
    //         printf("ex: %f\n", ex);
    //         printf("ex_: %f\n", ex_);
    //         printf("minus: %f\n", ex - ex_);
    //         printf("sum: %f\n", ex + ex_);

    //         printf("tanh: %f\n", tanh_res);
    //         printf("res: %f\n", res);
    //         printf("temp: %f\n", temp);
    //     }
    //     */
    // }else if(activation == "relu") {
    //     temp = temp > 0 ? temp : 0 ;
    // }
    
    temp = temp > 0 ? temp : 0 ;

    __syncthreads();
    output[offset_write] = __float2half(temp);   
}