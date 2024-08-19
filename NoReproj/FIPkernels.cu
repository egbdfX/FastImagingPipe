#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <cufft.h>

#define M_PI 3.14159265358979323846

/* The gridding kernels are developed based on SKA SDP (https://gitlab.com/ska-telescope/sdp/ska-sdp-func). */

__constant__ float quadrature_nodes[14] = {
	0.9964425,0.98130317,0.95425928,0.91563303,0.86589252,
	0.80564137,0.73561088,0.65665109,0.56972047,0.47587422,
	0.37625152,0.27206163,0.16456928,0.05507929
};
__constant__ float quadrature_weights[14] = {
	0.00912428,0.02113211,0.03290143,0.04427293,0.05510735,
	0.06527292,0.07464621,0.08311342,0.09057174,0.09693066,
	0.10211297,0.10605577,0.10871119,0.11004701
};
__constant__ float quadrature_kernel[14] = {
	7.71381676e-07,4.06901586e-06,2.09164257e-05,1.01923695e-04,
	4.61199576e-04,1.90183990e-03,7.02391280e-03,2.28652529e-02,
	6.46725327e-02,1.56933676e-01,3.23208771e-01,5.60024174e-01,
	8.10934691e-01,9.76937533e-01
};

float computeCeil(float num) {
    if (num<0) {
		return -floorf(-num);
	} else {
		return ceilf(num);
	}
}

float computeFloor(float num) {
    if (num<0) {
		return -ceilf(-num);
	} else {
		return floorf(num);
	}
}

__device__ float ceil_device(float num) {
    if (num<0) {
		return -floorf(-num);
	} else {
		return ceilf(num);
	}
}

__device__ float floor_device(float num) {
    if (num<0) {
		return -ceilf(-num);
	} else {
		return floorf(num);
	}
}

__device__ float exp_semicircle(const float beta, float x){
    const float xx = x*x;
    
    return ((xx > float(1.0)) ? float(0.0) : exp(beta*(sqrt(float(1.0) - xx) - float(1.0))));
}

__global__ void convolveKernel(float *conv_corr_kernel, size_t image_size, size_t grid_size, float conv_corr_norm_factor) {
    const int support = 8;
    size_t t1_t2 = blockIdx.x * blockDim.x + threadIdx.x;
    if (t1_t2 < image_size / 2 + 1) {
        float t1_t2_norm = static_cast<float>(t1_t2) / grid_size;
        float correction = 0.0;
        float angle;
        for (int i = 0; i < 14; ++i) {
            angle = M_PI * t1_t2_norm * support * quadrature_nodes[i];
            correction += quadrature_kernel[i] * quadrature_weights[i] * cosf(angle);
        }
        conv_corr_kernel[t1_t2] = correction * support / conv_corr_norm_factor;
    }
}

__global__ void computeVisWeighted(float *Vis_real, float *Vis_imag, size_t num_baselines, float inten_scale) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_baselines) {
		Vis_real[idx] = Vis_real[idx]/inten_scale;
		Vis_imag[idx] = Vis_imag[idx]/inten_scale;
	}
}

__global__ void gridding(float* B_in, float* w_grid_stack_real, float* w_grid_stack_imag, float* Vis_real, float* Vis_imag, float freq_hz, float uv_scale, size_t grid_size, size_t num_baselines) {
	float inv_wavelength = freq_hz / 299792458;
	const int support = 8;
	int half_support = support / 2;
	float inv_half_support = 1 / static_cast<float>(half_support);
    long int grid_min_uv = -static_cast<long int>(grid_size) / 2;
    long int grid_max_uv = (static_cast<long int>(grid_size) - 1) / 2;
    long int origin_offset_uv = static_cast<long int>(grid_size) / 2;
    const int KERNEL_SUPPORT_BOUND = 16;
    const float beta = 15.3704324328;
    float kernel_value;
	
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (idx < num_baselines) {
		float pos_u = B_in[idx*2+0] * inv_wavelength * uv_scale;
		float pos_v = B_in[idx*2+1] * inv_wavelength * uv_scale;
		long int grid_u_min = max(static_cast<long int>(ceil_device(pos_u - half_support)), grid_min_uv);
        long int grid_u_max = min(static_cast<long int>(floor_device(pos_u + half_support)), grid_max_uv);
        long int grid_v_min = max(static_cast<long int>(ceil_device(pos_v - half_support)), grid_min_uv);
        long int grid_v_max = min(static_cast<long int>(floor_device(pos_v + half_support)), grid_max_uv);
        if (grid_u_min > grid_u_max || grid_v_min > grid_v_max) {
            return;
        }
		float kernel_u[KERNEL_SUPPORT_BOUND], kernel_v[KERNEL_SUPPORT_BOUND];
		for (long int grid_u = grid_u_min; grid_u <= grid_u_max; grid_u++)
		{
			kernel_u[grid_u - grid_u_min] = exp_semicircle(beta,(static_cast<float>(grid_u) - pos_u) * inv_half_support);
		}
		for (long int grid_v = grid_v_min; grid_v <= grid_v_max; grid_v++)
		{
			kernel_v[grid_v - grid_v_min] = exp_semicircle(beta,(static_cast<float>(grid_v) - pos_v) * inv_half_support);
		}
		
        for (long int grid_u = grid_u_min; grid_u <= grid_u_max; grid_u++)
        {
            for (long int grid_v = grid_v_min; grid_v <= grid_v_max; grid_v++)
            {
                kernel_value = kernel_u[grid_u - grid_u_min] * kernel_v[grid_v - grid_v_min];
                if (((grid_u + grid_v) & 1) != 0) {
					kernel_value = -kernel_value;
				}
                const long int grid_offset_uvw = (grid_u + origin_offset_uv) * static_cast<long int>(grid_size) + (grid_v + origin_offset_uv);
                        
                atomicAdd(&w_grid_stack_real[grid_offset_uvw],Vis_real[idx] * kernel_value);
                atomicAdd(&w_grid_stack_imag[grid_offset_uvw],Vis_imag[idx] * kernel_value);
                }
		}
	}
}

__global__ void combineToComplex(float* w_real, float* w_imag, cufftComplex* complex_data, size_t grid_size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t size = grid_size * grid_size;
    if (idx < size) {
        complex_data[idx].x = w_real[idx];
        complex_data[idx].y = w_imag[idx];
    }
}

__global__ void ifftShift(cufftComplex* data, cufftComplex* data_shifted, size_t NX, size_t NY) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < NX && idy < NY) {
        size_t new_x = (idx + NX / 2) % NX;
        size_t new_y = (idy + NY / 2) % NY;
        size_t old_id = idy * NX + idx;
        size_t new_id = new_y * NX + new_x;
        
        data_shifted[new_id] = data[old_id];
    }
}

__global__ void accumulation(float* dirty_pre, cufftComplex* w_grid_stack_shifted, size_t image_size, size_t grid_size) {
	size_t half_image_size = image_size / 2;
	size_t grid_index_offset_image_centre = grid_size*grid_size/2 + grid_size/2;
	size_t image_index_offset_image_centre = half_image_size*image_size + half_image_size;
	long int idx = blockIdx.x * blockDim.x + threadIdx.x;
    long int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < image_size && idy < image_size) { 
		idx = idx - half_image_size;
		idy = idy - half_image_size;
		float pixel_sum = w_grid_stack_shifted[grid_index_offset_image_centre + idy*grid_size + idx].x;
        if (((abs(idx)+abs(idy)) & 1) != 0) {
			pixel_sum = - pixel_sum;
		}
		dirty_pre[image_index_offset_image_centre + idy*image_size + idx] += pixel_sum;
    }
}

__global__ void scaling(float* dirty_pre, float* conv_corr_kernel, size_t image_size, float conv_corr_norm_factor) {
	size_t half_image_size = image_size / 2;
	size_t image_index_offset_image_centre = half_image_size*image_size + half_image_size;
	long int idx = blockIdx.x * blockDim.x + threadIdx.x;
    long int idy = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (idx < image_size && idy < image_size) { 
		idx = idx - half_image_size;
		idy = idy - half_image_size;
        
        dirty_pre[image_index_offset_image_centre + idy * image_size + idx] *= 1/(conv_corr_kernel[abs(idx)]*conv_corr_kernel[abs(idy)]*conv_corr_norm_factor*conv_corr_norm_factor);
		dirty_pre[image_index_offset_image_centre + idy * image_size + idx] = abs(dirty_pre[image_index_offset_image_centre + idy * image_size + idx]);
	}
}

__global__ void coordschange(float* output_index, float* V_in, size_t image_size) {
	long int half_image_size = (long int)image_size / 2;
	
	long int idx = blockIdx.x * blockDim.x + threadIdx.x;
    long int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < image_size && idy < image_size) {
        output_index[(idx*image_size+idy)*2+0] = (-V_in[0*3+0]*(idx - half_image_size)+V_in[1*3+0]*(idy - half_image_size))/abs(V_in[2*3+2]) + half_image_size;
        output_index[(idx*image_size+idy)*2+1] = (-V_in[0*3+1]*(idx - half_image_size)+V_in[1*3+1]*(idy - half_image_size))/abs(V_in[2*3+2]) + half_image_size;
	}
}

__global__ void finalinterp(float* output_index, float* dirty_pre, float* dirty, size_t image_size) {
	long int idx = blockIdx.x * blockDim.x + threadIdx.x;
    long int idy = blockIdx.y * blockDim.y + threadIdx.y;
    size_t half_image_size = image_size / 2;
    size_t image_index_offset_image_centre = half_image_size*image_size + half_image_size;
    
    if (idx < image_size && idy < image_size) {
		float LL = output_index[(idx*image_size+idy)*2+0] - half_image_size;
		float MM = output_index[(idx*image_size+idy)*2+1] - half_image_size;
		
		idx = idx - half_image_size;
		idy = idy - half_image_size;
		
		if (abs(LL) < half_image_size-1 && abs(MM)<half_image_size-1) {
			atomicAdd(
				&dirty[
					static_cast<long int>(
						image_index_offset_image_centre+floor_device(MM)*image_size+floor_device(LL)
					)
				],
				(1-LL+floor_device(LL))*(1-MM+floor_device(MM))*
					dirty_pre[
						static_cast<long int>(
							image_index_offset_image_centre+idy*image_size+idx
						)
					]
			);
            atomicAdd(
				&dirty[
					static_cast<long int>(
						image_index_offset_image_centre+ceil_device(MM)*image_size+floor_device(LL)
					)
				],
				(1-LL+floor_device(LL))*(MM-floor_device(MM))*
					dirty_pre[
						static_cast<long int>(
							image_index_offset_image_centre+idy*image_size+idx
						)
					]
			);
            atomicAdd(
				&dirty[
					static_cast<long int>(
						image_index_offset_image_centre+floor_device(MM)*image_size+ceil_device(LL)
					)
				],
				(LL-floor_device(LL))*(1-MM+floor_device(MM))*
					dirty_pre[
						static_cast<long int>(
							image_index_offset_image_centre+idy*image_size+idx
						)
					]
			);
            atomicAdd(
				&dirty[
					static_cast<long int>(
						image_index_offset_image_centre+ceil_device(MM)*image_size+ceil_device(LL)
					)
				],
				(LL-floor_device(LL))*(MM-floor_device(MM))*
					dirty_pre[
						static_cast<long int>(
							image_index_offset_image_centre+idy*image_size+idx
						)
					]
			);
		}
	}
}

int FIpipe(float* Visreal, float* Visimag, float* Bin, float* Vin, float* dirty_image, size_t num_baselines, size_t image_size, float freq_hz, float cell_size){
	float* Vis_real;
	float* Vis_imag;
	float* B_in;
	float* V_in;
	float* dirty;
	float* dirty_pre;
	float* conv_corr_kernel;
	float* w_grid_stack_real;
	float* w_grid_stack_imag;
	float* pixel_ind;
	cudaError_t cudaStatus;
	cufftComplex* w_grid_stack;
	cufftComplex* w_grid_stack_shifted;
	float* output_index;
	cudaError_t cudaError;
	double *h_output_index = new double[image_size * image_size * 2];
    for (size_t i = 0; i < image_size * image_size * 2; i++) {
        h_output_index[i] = 0.0;
    }
	
	cudaEvent_t start, stop, eventstream;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&eventstream);
	
	size_t grid_size = computeCeil(1.5*static_cast<float>(image_size));
	float uv_scale = cell_size*grid_size;
	float conv_corr_norm_factor = 2.4937047051153827;
	
	cudaMalloc((void**)&Vis_real, num_baselines * 1 * sizeof(float));
	cudaMalloc((void**)&Vis_imag, num_baselines * 1 * sizeof(float));
	cudaMalloc((void**)&B_in, num_baselines * 2 * sizeof(float));
	cudaMalloc((void**)&V_in, 3 * 3 * sizeof(float));
	cudaMalloc((void**)&dirty, image_size * image_size * sizeof(float));
	cudaMalloc((void**)&dirty_pre, image_size * image_size * sizeof(float));
	cudaMalloc((void**)&conv_corr_kernel, (image_size/2+1)*sizeof(float));
	cudaMalloc((void**)&w_grid_stack_real, grid_size * grid_size * sizeof(float));
	cudaMalloc((void**)&w_grid_stack_imag, grid_size * grid_size * sizeof(float));
	cudaMalloc((void**)&w_grid_stack, grid_size * grid_size * sizeof(cufftComplex));
	cudaMalloc((void**)&w_grid_stack_shifted, grid_size * grid_size * sizeof(cufftComplex));
	cudaMalloc((void**)&output_index, image_size * image_size * 2 * sizeof(float));
	cudaMalloc((void**)&pixel_ind, image_size * image_size * 2 * sizeof(float));
	
	cudaMemcpy(Vis_real, Visreal, num_baselines * 1 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Vis_imag, Visimag, num_baselines * 1 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(B_in, Bin, num_baselines * 2 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(V_in, Vin, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice); // cross term included
	cudaMemcpy(dirty, dirty_image, image_size * image_size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemset(dirty_pre, 0.0, image_size * image_size * sizeof(float));
	cudaMemset(conv_corr_kernel, 0.0, (image_size/2+1) * sizeof(float));
	cudaMemset(w_grid_stack_real, 0.0, grid_size * grid_size * sizeof(float));
	cudaMemset(w_grid_stack_imag, 0.0, grid_size * grid_size * sizeof(float));
	cudaMemset(output_index, 0.0, image_size * image_size * 2 * sizeof(float));

	cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Error 1 : %s\n", cudaGetErrorString(cudaStatus));
	}
	cudaError = cudaGetLastError();
	if(cudaError != cudaSuccess){
		printf("ERROR! GPU Kernel 1 error.\n");
		printf("CUDA error code: %d; string: %s;\n", (int) cudaError, cudaGetErrorString(cudaError));
	}
	else {
		printf("No CUDA error 1.\n");
	}
	cudaStream_t stream1, stream2;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);
	
	cudaEventRecord(start);
	/* ****************************************************** */
	size_t num_threads = 1024;
	size_t num_blocks = computeCeil(static_cast<float>(image_size/2+1)/num_threads);
	convolveKernel<<<num_blocks,num_threads,0,stream2>>>(conv_corr_kernel, image_size, grid_size, conv_corr_norm_factor);
	cudaError = cudaGetLastError();
	if(cudaError != cudaSuccess){
		printf("ERROR! GPU Kernel 2 error.\n");
		printf("CUDA error code: %d; string: %s;\n", (int) cudaError, cudaGetErrorString(cudaError));
	}
	else {
		printf("No CUDA error 2.\n");
	}
	
	float inten_scale = std::abs(Vin[0*3+0]*Vin[1*3+1]-Vin[0*3+1]*Vin[1*3+0]);
	
	/* ****************************************************** */
	num_threads = 1024;
	num_blocks = computeCeil(static_cast<float>(num_baselines)/num_threads);
	computeVisWeighted<<<num_blocks,num_threads,0,stream1>>>(Vis_real,Vis_imag,num_baselines,inten_scale);
	cudaError = cudaGetLastError();
	if(cudaError != cudaSuccess){
		printf("ERROR! GPU Kernel 3 error.\n");
		printf("CUDA error code: %d; string: %s;\n", (int) cudaError, cudaGetErrorString(cudaError));
	}
	else {
		printf("No CUDA error 3.\n");
	}
	
	/* ****************************************************** */
	num_threads = 1024;
	num_blocks = computeCeil(static_cast<float>(num_baselines)/num_threads);
	gridding<<<num_blocks,num_threads,0,stream1>>>(B_in, w_grid_stack_real, w_grid_stack_imag, Vis_real, Vis_imag, freq_hz, uv_scale, grid_size, num_baselines);
	cudaError = cudaGetLastError();
	if(cudaError != cudaSuccess){
		printf("ERROR! GPU Kernel 4 error.\n");
		printf("CUDA error code: %d; string: %s;\n", (int) cudaError, cudaGetErrorString(cudaError));
	}
	else {
		printf("No CUDA error 4.\n");
	}
	
	/* ****************************************************** */
	num_threads = 1024;
	num_blocks = computeCeil(static_cast<float>(grid_size * grid_size)/num_threads);
	combineToComplex<<<num_blocks,num_threads,0,stream1>>>(w_grid_stack_real, w_grid_stack_imag, w_grid_stack, grid_size);
	cudaError = cudaGetLastError();
	if(cudaError != cudaSuccess){
		printf("ERROR! GPU Kernel 5 error.\n");
		printf("CUDA error code: %d; string: %s;\n", (int) cudaError, cudaGetErrorString(cudaError));
	}
	else {
		printf("No CUDA error 5.\n");
	}
	
	/* ****************************************************** */
	num_threads = 32;
	dim3 numThreads(num_threads, num_threads);
	dim3 numBlocks(computeCeil(static_cast<float>(grid_size)/num_threads), computeCeil(static_cast<float>(grid_size)/num_threads));
    ifftShift<<<numBlocks,numThreads,0,stream1>>>(w_grid_stack, w_grid_stack_shifted, grid_size, grid_size);
	cudaError = cudaGetLastError();
	if(cudaError != cudaSuccess){
		printf("ERROR! GPU Kernel 6 error.\n");
		printf("CUDA error code: %d; string: %s;\n", (int) cudaError, cudaGetErrorString(cudaError));
	}
	else {
		printf("No CUDA error 6.\n");
	}
    
    /* ****************************************************** */
    cufftHandle plan;
    cufftCreate(&plan);
	cufftSetStream(plan, stream1);
    cufftPlan2d(&plan, grid_size, grid_size, CUFFT_C2C);

    cufftExecC2C(plan, w_grid_stack_shifted, w_grid_stack_shifted, CUFFT_INVERSE);
	cudaError = cudaGetLastError();
	if(cudaError != cudaSuccess){
		printf("ERROR! GPU Kernel 7 error.\n");
		printf("CUDA error code: %d; string: %s;\n", (int) cudaError, cudaGetErrorString(cudaError));
	}
	else {
		printf("No CUDA error 7.\n");
	}
	
	/* ****************************************************** */
	numThreads.x = num_threads;
	numThreads.y = num_threads;
    numBlocks.x = computeCeil(static_cast<float>(image_size)/num_threads);
    numBlocks.y = computeCeil(static_cast<float>(image_size)/num_threads);
    accumulation<<<numBlocks,numThreads,0,stream1>>>(dirty_pre, w_grid_stack_shifted, image_size, grid_size);
	cudaError = cudaGetLastError();
	if(cudaError != cudaSuccess){
		printf("ERROR! GPU Kernel 8 error.\n");
		printf("CUDA error code: %d; string: %s;\n", (int) cudaError, cudaGetErrorString(cudaError));
	}
	else {
		printf("No CUDA error 8.\n");
	}
    
    /* ****************************************************** */
    numThreads.x = num_threads;
    numThreads.y = num_threads;
    numBlocks.x = computeCeil(static_cast<float>(image_size)/num_threads);
    numBlocks.y = computeCeil(static_cast<float>(image_size)/num_threads);
    scaling<<<numBlocks,numThreads,0,stream1>>>(dirty_pre, conv_corr_kernel, image_size, conv_corr_norm_factor);
	cudaError = cudaGetLastError();
	if(cudaError != cudaSuccess){
		printf("ERROR! GPU Kernel 9 error.\n");
		printf("CUDA error code: %d; string: %s;\n", (int) cudaError, cudaGetErrorString(cudaError));
	}
	else {
		printf("No CUDA error 9.\n");
	}
	
	/* ****************************************************** */
	numThreads.x = num_threads;
	numThreads.y = num_threads;
    numBlocks.x = computeCeil(static_cast<float>(image_size)/num_threads);
    numBlocks.y = computeCeil(static_cast<float>(image_size)/num_threads);
	coordschange<<<numBlocks,numThreads,0,stream2>>>(output_index, V_in, image_size);
	cudaError = cudaGetLastError();
	if(cudaError != cudaSuccess){
		printf("ERROR! GPU Kernel 10 error.\n");
		printf("CUDA error code: %d; string: %s;\n", (int) cudaError, cudaGetErrorString(cudaError));
	}
	else {
		printf("No CUDA error 10.\n");
	}
	cudaEventRecord(eventstream,stream2);
	
	cudaStreamWaitEvent(stream1,eventstream,0);
	
	/* ****************************************************** */
	numThreads.x = num_threads;
	numThreads.y = num_threads;
    numBlocks.x = computeCeil(static_cast<float>(image_size)/num_threads);
    numBlocks.y = computeCeil(static_cast<float>(image_size)/num_threads);
    finalinterp<<<numBlocks,numThreads,0,stream1>>>(output_index, dirty_pre, dirty, image_size);
	cudaError = cudaGetLastError();
	if(cudaError != cudaSuccess){
		printf("ERROR! GPU Kernel 14 error.\n");
		printf("CUDA error code: %d; string: %s;\n", (int) cudaError, cudaGetErrorString(cudaError));
	}
	else {
		printf("No CUDA error 14.\n");
	}
	
	cudaStreamSynchronize(stream1);
	
	cudaEventDestroy(eventstream);
	cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    
    cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout << "Time elapsed: " << milliseconds << " ms" << std::endl;
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
    
    cudaMemcpy(dirty_image, dirty, image_size * image_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Error 15 : %s\n", cudaGetErrorString(cudaStatus));
	}
	cudaError = cudaGetLastError();
	if(cudaError != cudaSuccess){
		printf("ERROR! GPU Kernel 15 error.\n");
		printf("CUDA error code: %d; string: %s;\n", (int) cudaError, cudaGetErrorString(cudaError));
	}
	else {
		printf("No CUDA error 15.\n");
	}
	
	cudaFree(Vis_real);
	cudaFree(Vis_imag);
	cudaFree(B_in);
	cudaFree(V_in);
	cudaFree(dirty);
	cudaFree(dirty_pre);
	cudaFree(conv_corr_kernel);
	cudaFree(w_grid_stack_real);
	cudaFree(w_grid_stack_imag);
	cudaFree(w_grid_stack);
	cudaFree(w_grid_stack_shifted);
	cudaFree(output_index);
	
	return 0;
}
	
