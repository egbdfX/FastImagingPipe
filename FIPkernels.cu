#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <cufft.h>
#include <device_launch_parameters.h>
#include <math_constants.h>

#define M_PI 3.14159265358979323846
#define WCSTRIG_TOL 1e-10

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

long int computeCeil(float num) {
	if (num<0) {
		return -floorf(-num);
	} else {
		return ceilf(num);
	}
}

long int computeFloor(float num) {
	if (num<0) {
		return -ceilf(-num);
	} else {
		return floorf(num);
	}
}

__device__ long int ceil_device(float num) {
	if (num<0) {
		return -floorf(-num);
	} else {
		return ceilf(num);
	}
}

__device__ long int floor_device(float num) {
	if (num<0) {
		return -ceilf(-num);
	} else {
		return floorf(num);
	}
}

__device__ float fmod_device(float x, float y) {
	return fmod(x, y);
}

__device__ float exp_semicircle(const float beta, float x){
	const float xx = x*x;
    
	return ((xx > float(1.0)) ? float(0.0) : exp(beta*(sqrt(float(1.0) - xx) - float(1.0))));
}

__device__ void __sincosd(float angle, float &s, float &c) {
	// angle in degrees
	if (fmod(angle, 90.0f) == 0) {
		int i = static_cast<int>(fabsf(floor_device(angle / 90.0f + 0.5f))) % 4;
		switch (i) {
			case 0:
				s = 0.0f;
				c = 1.0f;
				break;
			case 1:
				s = (angle > 0.0f) ? 1.0f : -1.0f;
				c = 0.0f;
				break;
			case 2:
				s = 0.0f;
				c = -1.0f;
				break;
			case 3:
				s = (angle > 0.0f) ? -1.0f : 1.0f;
				c = 0.0f;
				break;
		}
	} else {
		s = sinf(angle * M_PI / 180.0f);
		c = cosf(angle * M_PI / 180.0f);
	}
}

__device__ float __sind(float angle) {
	// angle in degrees
	if (fmod(angle, 90.0f) == 0) {
		int i = static_cast<int>(fabsf(floor_device(angle / 90.0f - 0.5f))) % 4;
		switch (i) {
			case 0:
				return 1.0f;
			case 1:
				return 0.0f;
			case 2:
				return -1.0f;
			case 3:
				return 0.0f;
		}
	} else {
		return sinf(angle * M_PI / 180.0f);
	}
}

__device__ float __cosd(float angle) {
	// angle in degrees
	if (fmod(angle, 90.0f) == 0) {
		int i = static_cast<int>(fabsf(floor_device(angle / 90.0f + 0.5f))) % 4;
		switch (i) {
			case 0:
				return 1.0f;
			case 1:
				return 0.0f;
			case 2:
				return -1.0f;
			case 3:
				return 0.0f;
		}
	} else {
		return cosf(angle * M_PI / 180.0f);
	}
}

__device__ float __atan2d(float y, float x) {
	if (y == 0.0f) {
		return (x >= 0.0f) ? 0.0f : 180.0f;
	} else if (x == 0.0f) {
		return (y > 0.0f) ? 90.0f : -90.0f;
	} else {
		return atan2f(y, x) * 180.0f / M_PI;
	}
}

__device__ float __acosd(float v) {
	if (v >= 1.0f && v - 1.0f < WCSTRIG_TOL) {
		return 0.0f;
	} else if (v == 0.0f) {
		return 90.0f;
	} else if (v <= -1.0f && v + 1.0f > -WCSTRIG_TOL) {
		return 180.0f;
	} else {
		return acosf(v) * 180.0f / M_PI;
	}
}

__device__ float __asind(float v) {
	if (v <= -1.0f && v + 1.0f > -WCSTRIG_TOL) {
		return -90.0f;
	} else if (v == 0.0f) {
		return 0.0f;
	} else if (v >= 1.0f && v - 1.0f < WCSTRIG_TOL) {
		return 90.0f;
	} else {
		return asinf(v) * 180.0f / M_PI;
	}
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

__global__ void computeVisWeighted(float *Vis_real, float *Vis_imag, size_t num_baselines, float* V_in) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	float inten_scale = fabs(V_in[0*3+0]*V_in[1*3+1]-V_in[0*3+1]*V_in[1*3+0]);
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
		long int grid_u_min = max(ceil_device(pos_u - half_support), grid_min_uv);
		long int grid_u_max = min(floor_device(pos_u + half_support), grid_max_uv);
		long int grid_v_min = max(ceil_device(pos_v - half_support), grid_min_uv);
		long int grid_v_max = min(floor_device(pos_v + half_support), grid_max_uv);
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
		dirty_pre[image_index_offset_image_centre + idy * image_size + idx] = fabs(dirty_pre[image_index_offset_image_centre + idy * image_size + idx]);
	}
}

__global__ void coordschange(float* output_index, float* V_in, size_t image_size) {
	size_t half_image_size = image_size / 2;
	
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
    
	if (idx < image_size && idy < image_size) {
        	output_index[(idx*image_size+idy)*2+0] = (-V_in[0*3+0]*(static_cast<float>(idx) - static_cast<float>(half_image_size))+V_in[1*3+0]*(static_cast<float>(idy) - static_cast<float>(half_image_size)))/fabs(V_in[2*3+2]) + static_cast<float>(half_image_size);
        	output_index[(idx*image_size+idy)*2+1] = (-V_in[0*3+1]*(static_cast<float>(idx) - static_cast<float>(half_image_size))+V_in[1*3+1]*(static_cast<float>(idy) - static_cast<float>(half_image_size)))/fabs(V_in[2*3+2]) + static_cast<float>(half_image_size);	
	}
}

__global__ void p2p(float* output_index, float* V_in, float dc, size_t di) {
	/* According to paper: M. R.  Calabretta, E. W.  Greisen, 'Representations of celestial coordinates in FITS,' A&A,395(3),1077-1122,2002.*/
	
	long int idx = blockIdx.x * blockDim.x + threadIdx.x;
	long int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
	float xi = V_in[6]/V_in[8];
	float eta = V_in[7]/V_in[8];
	dc = dc / M_PI * 180;
    
	if (idx < di && idy < di) {	
		float p1 = output_index[(idx*di+idy)*2+0]; // p2
		float p2 = output_index[(idx*di+idy)*2+1]; // p1
		
		float x = -static_cast<float>(dc) * (p1 - (static_cast<float>(di) / 2.0f + 1.0f));
		float y = static_cast<float>(dc) * (p2 - (static_cast<float>(di) / 2.0f + 1.0f));

		float r0 = 180.0f / M_PI;
		float x0 = x / r0;
		float y0 = y / r0;
		float r2 = x0 * x0 + y0 * y0;
		
		float phi;
		if (r2 != 0.0f) {
			phi = __atan2d(x0, -y0);
		} else {
			phi = 0.0f;
		}

		float theta;
		if (r2 < 0.5f) {
			theta = __acosd(sqrtf(r2));
		} else if (r2 <= 1.0f) {
			theta = __asind(sqrtf(1.0f - r2));
		} else {
			return;
		}

		float sinphi, cosphi;
		__sincosd(phi, sinphi, cosphi);
		x = sinphi;
		y = cosphi;

		float t = (90.0f - fabsf(theta)) / 180.0f * M_PI;
		float z, costhe;
		if (t < 1.0e-5f) {
			if (theta > 0.0f) {
				z = t * t / 2.0f;
			} else {
				z = 2.0f - t * t / 2.0f;
			}
			costhe = t;
		} else {
			z = 1.0f - __sind(theta);
			costhe = __cosd(theta);
		}

		r0 = 180.0f / M_PI;
		float r = r0 * costhe;
		float w = xi * xi + eta * eta;

		if (w == 0.0f) {
			x = r * x;
			y = -r * y;
		} else {
			z = z * r0;
			float z1 = xi * z;
			float z2 = eta * z;
			x = r * x + z1;
			y = -r * y + z2;
		}

		output_index[(idx*di+idy)*2+0] = -1.0f / static_cast<float>(dc) * x + static_cast<float>(di) / 2.0f + 1.0f;
		output_index[(idx*di+idy)*2+1] = 1.0f / static_cast<float>(dc) * y + static_cast<float>(di) / 2.0f + 1.0f;
	}
}

__global__ void finalinterp(float* output_index, float* dirty_pre, float* dirty, size_t image_size) {
	long int idx = blockIdx.x * blockDim.x + threadIdx.x;
	long int idy = blockIdx.y * blockDim.y + threadIdx.y;
	size_t half_image_size = image_size / 2;
	size_t image_index_offset_image_centre = static_cast<long int>(half_image_size*image_size + half_image_size);
    
	if (idx < image_size && idy < image_size) {
		float LL = output_index[(static_cast<size_t>(idx)*image_size+static_cast<size_t>(idy))*2+0] - static_cast<float>(half_image_size);
		float MM = output_index[(static_cast<size_t>(idx)*image_size+static_cast<size_t>(idy))*2+1] - static_cast<float>(half_image_size);
		
		idx = idx - static_cast<long int>(half_image_size);
		idy = idy - static_cast<long int>(half_image_size);
		
		if (fabs(LL) < half_image_size-1 && fabs(MM)<half_image_size-1) {
			atomicAdd(
				&dirty[image_index_offset_image_centre+floor_device(MM)*static_cast<long int>(image_size)+floor_device(LL)],
				(1-LL+floor_device(LL))*(1-MM+floor_device(MM))*
					dirty_pre[image_index_offset_image_centre+idy*static_cast<long int>(image_size)+idx]
			);
			atomicAdd(
				&dirty[image_index_offset_image_centre+ceil_device(MM)*static_cast<long int>(image_size)+floor_device(LL)],
				(1-LL+floor_device(LL))*(MM-floor_device(MM))*
					dirty_pre[image_index_offset_image_centre+idy*static_cast<long int>(image_size)+idx]
			);
			atomicAdd(
				&dirty[image_index_offset_image_centre+floor_device(MM)*static_cast<long int>(image_size)+ceil_device(LL)],
				(LL-floor_device(LL))*(1-MM+floor_device(MM))*
					dirty_pre[image_index_offset_image_centre+idy*static_cast<long int>(image_size)+idx]
			);
			atomicAdd(
				&dirty[image_index_offset_image_centre+ceil_device(MM)*static_cast<long int>(image_size)+ceil_device(LL)],
				(LL-floor_device(LL))*(MM-floor_device(MM))*
					dirty_pre[image_index_offset_image_centre+idy*static_cast<long int>(image_size)+idx]
			);
		}
	}
}

__global__ void setNonPositiveToC(float* restored, size_t size, float C) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
	if (idx < size) {
		restored[idx] = (restored[idx] <= 0) ? C : restored[idx];
	}
}

__global__ void subtraction(float* data_1, float* data_2, float* diff_out, size_t size) {
	size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    
	if (idx < size) {
		diff_out[idx] = fabs(data_1[idx] - data_2[idx]);
	}
}

__global__ void max_large(float* dirty, float* result, size_t region_num, size_t region_size, size_t ima) {

	extern  __shared__  float sharedNumDen[];
    
	size_t bid = blockIdx.x; // tile index
	size_t tid = threadIdx.x;
    
	size_t i_id = static_cast<size_t>(floor_device(bid/region_num));
	size_t j_id = fmod_device(bid,region_num);
	size_t factor = static_cast<size_t>(ceil_device(static_cast<float>(region_size*region_size)/1024.0f));
    
	size_t I_id;
	size_t J_id;
	size_t rows;
	size_t cols;
    
	for (size_t fac = 1; fac <= factor;fac = fac + 1){
		if (tid+(fac-1)*1024 < region_size*region_size){
			if (fac == 1) {
				sharedNumDen[tid] = 0; // Max
			}
			rows = static_cast<size_t>(floor_device((tid+(fac-1)*1024)/region_size));
			cols = fmod_device((tid+(fac-1)*1024),region_size);
			
			I_id = i_id * region_size + rows;
			J_id = j_id * region_size + cols;
			
			sharedNumDen[tid] = max(sharedNumDen[tid], dirty[I_id * ima + J_id]);
			} else {
			if (fac == 1) {
				sharedNumDen[tid] = 0; // Max
			}
		}
	}
    
	for (size_t d = blockDim.x/2;d>0;d = d/2){
		__syncthreads();
		if (tid<d) {
			sharedNumDen[tid] = max(sharedNumDen[tid], sharedNumDen[tid+d]);
		}
	}
	
	if (tid==0) {
		result[bid] = sharedNumDen[0];
	}
}

__global__ void max_small(float* max_tmp, float* maxall, size_t ima, int bid_ind) {
	
	extern  __shared__  float sharedNumDen[];
    
	size_t tid = threadIdx.x;
    
	sharedNumDen[tid] = max_tmp[tid];
	__syncthreads();
    
	for (size_t d = blockDim.x/2;d>0;d = d/2){
		if (tid<d) {
			sharedNumDen[tid] = max(sharedNumDen[tid], sharedNumDen[tid+d]);
		}
		__syncthreads();
	}
	
	if (tid==0) {
		maxall[bid_ind] = sharedNumDen[0];
	}
}

__global__ void tlisi(float* diff_out, float* snap, float* result, size_t unit_size, size_t ima, size_t unit_num, float* maxall) {

	extern  __shared__  float sharedNumDen[];
	float maxallval = max(maxall[0], max(maxall[1], maxall[2]));
	size_t bid = blockIdx.x; // tile index
	size_t tid = threadIdx.x;
    
	size_t i_id = static_cast<size_t>(floor_device(bid/unit_num));
	size_t j_id = fmod_device(bid,unit_num);
	size_t factor = static_cast<size_t>(ceil_device(static_cast<float>(unit_size*unit_size)/1024.0f));
    
	size_t I_id;
	size_t J_id;
	size_t rows;
	size_t cols;
    
	for (size_t fac = 1; fac <= factor;fac = fac + 1){
		if (tid+(fac-1)*1024 < unit_size*unit_size){
			if (fac == 1) {
				sharedNumDen[tid] = 0; // Sum of diff_out
				sharedNumDen[tid+1024] = 0; // Max of diff_out
				sharedNumDen[tid+2048] = 0; // Sum of r
			}
			rows = static_cast<size_t>(floor_device((tid+(fac-1)*1024)/unit_size));
			cols = fmod_device((tid+(fac-1)*1024),unit_size);
			
			I_id = i_id * unit_size + rows;
			J_id = j_id * unit_size + cols;
			
			sharedNumDen[tid] = sharedNumDen[tid] + diff_out[I_id * ima + J_id];
			sharedNumDen[tid+1024] = max(sharedNumDen[tid+1024], diff_out[I_id * ima + J_id]);
			sharedNumDen[tid+2048] = (diff_out[I_id * ima + J_id]/snap[I_id * ima + J_id] < 1) ? sharedNumDen[tid+2048] +  diff_out[I_id * ima + J_id]/snap[I_id * ima + J_id]: sharedNumDen[tid+2048] + 1;
		} else {
			if (fac == 1) {
				sharedNumDen[tid] = 0; // Sum of diff_out
				sharedNumDen[tid+1024] = 0; // Max of diff_out
				sharedNumDen[tid+2048] = 0; // Sum of r
			}
		}
	}
    
	for (size_t d = blockDim.x/2;d>0;d = d/2){
		__syncthreads();
		if (tid<d) {
			sharedNumDen[tid] += sharedNumDen[tid+d];
			sharedNumDen[tid+1024] = max(sharedNumDen[tid+1024], sharedNumDen[tid+1024+d]);
			sharedNumDen[tid+2048] += sharedNumDen[tid+2048+d];
		}
	}
	
	if (tid==0) {
		result[bid] = 1-(sharedNumDen[0]/unit_size/unit_size)*sharedNumDen[1024]*(sharedNumDen[2048]/unit_size/unit_size)/maxallval/maxallval;
	}
}

int FIpipe(float* Visreal, float* Visimag, float* Bin, float* Vin, 
           float* result_array,
           size_t num_baselines, size_t image_size, size_t num_snapshots,
           float freq_hz, float cell_size,
           size_t unit_size){
	float *Vis_real, *Vis_imag, *B_in, *V_in;
	float *Vis_realtmp, *Vis_imagtmp, *B_intmp, *V_intmp;
	float *pinned_Vis_real, *pinned_Vis_imag, *pinned_B_in, *pinned_V_in;
	float *dirty1, *dirty2, *dirty3;
	float *dirty_pre, *conv_corr_kernel, *w_grid_stack_real, *w_grid_stack_imag, *pixel_ind, *output_index, *max_tmp, *maxall;
	cudaError_t cudaStatus,cudaError;
	cufftComplex *w_grid_stack, *w_grid_stack_shifted;
	int bid_ind;
	size_t shared_mem_size;
	size_t region_num = 32;
	size_t region_size = static_cast<size_t>(computeCeil(static_cast<float>(image_size)/static_cast<float>(region_num)));
	
	cudaEvent_t start, stop, eventstream[3], events[3], events_kernel[3];
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&eventstream[0]);
	cudaEventCreate(&eventstream[1]);
	cudaEventCreate(&eventstream[2]);
	cudaEventCreate(&events[0]);
	cudaEventCreate(&events[1]);
	cudaEventCreate(&events[2]);
	cudaEventCreate(&events_kernel[0]);
	cudaEventCreate(&events_kernel[1]);
	cudaEventCreate(&events_kernel[2]);
	
	size_t grid_size = static_cast<size_t>(computeCeil(1.5*static_cast<float>(image_size)));
	float uv_scale = cell_size*grid_size;
	float conv_corr_norm_factor = 2.4937047051153827;
	
	cudaMalloc((void**)&dirty1, image_size * image_size * sizeof(float));
	cudaMalloc((void**)&dirty2, image_size * image_size * sizeof(float));
	cudaMalloc((void**)&dirty3, image_size * image_size * sizeof(float));
	cudaMalloc((void**)&dirty_pre, image_size * image_size * sizeof(float));
	cudaMalloc((void**)&conv_corr_kernel, (image_size/2+1)*sizeof(float));
	cudaMalloc((void**)&w_grid_stack_real, grid_size * grid_size * sizeof(float));
	cudaMalloc((void**)&w_grid_stack_imag, grid_size * grid_size * sizeof(float));
	cudaMalloc((void**)&w_grid_stack, grid_size * grid_size * sizeof(cufftComplex));
	cudaMalloc((void**)&w_grid_stack_shifted, grid_size * grid_size * sizeof(cufftComplex));
	cudaMalloc((void**)&output_index, image_size * image_size * 2 * sizeof(float));
	cudaMalloc((void**)&pixel_ind, image_size * image_size * 2 * sizeof(float));
	cudaMalloc((void**)&max_tmp, region_num * region_num * sizeof(float));
	cudaMalloc((void**)&maxall, 3 * sizeof(float));
	cudaMalloc((void**)&Vis_realtmp, num_baselines * 1 * sizeof(float));
	cudaMalloc((void**)&Vis_imagtmp, num_baselines * 1 * sizeof(float));
	cudaMalloc((void**)&B_intmp, num_baselines * 2 * sizeof(float));
	cudaMalloc((void**)&V_intmp, 3 * 3 * sizeof(float));
	cudaMalloc((void**)&Vis_real, num_baselines * 1 * sizeof(float));
	cudaMalloc((void**)&Vis_imag, num_baselines * 1 * sizeof(float));
	cudaMalloc((void**)&B_in, num_baselines * 2 * sizeof(float));
	cudaMalloc((void**)&V_in, 3 * 3 * sizeof(float));
	
	cudaMallocHost((void**)&pinned_Vis_real, num_baselines*num_snapshots*sizeof(float));
	cudaMallocHost((void**)&pinned_Vis_imag, num_baselines*num_snapshots*sizeof(float));
	cudaMallocHost((void**)&pinned_B_in, num_baselines*2*num_snapshots*sizeof(float));
	cudaMallocHost((void**)&pinned_V_in, 3*3*num_snapshots*sizeof(float));
	memcpy(pinned_Vis_real, Visreal, num_baselines*num_snapshots*sizeof(float));
	memcpy(pinned_Vis_imag, Visimag, num_baselines*num_snapshots*sizeof(float));
	memcpy(pinned_B_in, Bin, num_baselines*2*num_snapshots*sizeof(float));
	memcpy(pinned_V_in, Vin, 3*3*num_snapshots*sizeof(float));
	
	cudaMemset(dirty1, 0, image_size * image_size * sizeof(float));
	cudaMemset(dirty2, 0, image_size * image_size * sizeof(float));
	cudaMemset(dirty3, 0, image_size * image_size * sizeof(float));
	
	cudaError = cudaGetLastError();
	if(cudaError != cudaSuccess){
		printf("ERROR! GPU Kernel 1 error.\n");
		printf("CUDA error code: %d; string: %s;\n", (int) cudaError, cudaGetErrorString(cudaError));
	}
	else {
		printf("No CUDA error 1.\n");
	}
	
	cudaStream_t stream1, stream2, stream_pcie;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);
	cudaStreamCreate(&stream_pcie);
	
	cufftHandle plan;
	cufftCreate(&plan);
	cufftSetStream(plan, stream1);
	cufftPlan2d(&plan, grid_size, grid_size, CUFFT_C2C);
	
	size_t num_threads;
	size_t num_blocks;
	dim3 numThreads;
	dim3 numBlocks;
	
	cudaEventRecord(start);
	/* ****************************************************** */
	for (int ind = 0; ind < 3; ++ind){
		cudaMemcpyAsync(Vis_realtmp, pinned_Vis_real+static_cast<size_t>(ind)*num_baselines, num_baselines * 1 * sizeof(float), cudaMemcpyHostToDevice, stream_pcie);
		cudaMemcpyAsync(Vis_imagtmp, pinned_Vis_imag+static_cast<size_t>(ind)*num_baselines, num_baselines * 1 * sizeof(float), cudaMemcpyHostToDevice, stream_pcie);
		cudaMemcpyAsync(B_intmp, pinned_B_in+static_cast<size_t>(ind)*num_baselines*2, num_baselines * 2 * sizeof(float), cudaMemcpyHostToDevice, stream_pcie);
		cudaMemcpyAsync(V_intmp, pinned_V_in+static_cast<size_t>(ind)*9, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice, stream_pcie); // cross term included
	    
		if (ind == 0) {
			cudaMemcpyAsync(Vis_real, Vis_realtmp, num_baselines * 1 * sizeof(float), cudaMemcpyDeviceToDevice, stream_pcie);
			cudaMemcpyAsync(Vis_imag, Vis_imagtmp, num_baselines * 1 * sizeof(float), cudaMemcpyDeviceToDevice, stream_pcie);
			cudaMemcpyAsync(B_in, B_intmp, num_baselines * 2 * sizeof(float), cudaMemcpyDeviceToDevice, stream_pcie);
			cudaMemcpyAsync(V_in, V_intmp, 3 * 3 * sizeof(float), cudaMemcpyDeviceToDevice, stream_pcie);
			cudaEventRecord(events[ind],stream_pcie);
		}

		else{
			cudaStreamWaitEvent(stream1,events[ind-1],0);
			cudaStreamWaitEvent(stream2,events[ind-1],0);
			
			cudaMemsetAsync(dirty_pre, 0, image_size * image_size * sizeof(float),stream1);
			cudaMemsetAsync(conv_corr_kernel, 0, (image_size/2+1) * sizeof(float),stream2);
			cudaMemsetAsync(w_grid_stack_real, 0, grid_size * grid_size * sizeof(float),stream1);
			cudaMemsetAsync(w_grid_stack_imag, 0, grid_size * grid_size * sizeof(float),stream1);
			cudaMemsetAsync(output_index, 0, image_size * image_size * 2 * sizeof(float),stream2);
			
			num_threads = 1024;
			num_blocks = computeCeil(static_cast<float>(image_size/2+1)/num_threads);
			convolveKernel<<<num_blocks,num_threads,0,stream2>>>(conv_corr_kernel, image_size, grid_size, conv_corr_norm_factor);
			num_blocks = computeCeil(static_cast<float>(num_baselines)/num_threads);
			computeVisWeighted<<<num_blocks,num_threads,0,stream1>>>(Vis_real,Vis_imag,num_baselines,V_in);
			gridding<<<num_blocks,num_threads,0,stream1>>>(B_in, w_grid_stack_real, w_grid_stack_imag, Vis_real, Vis_imag, freq_hz, uv_scale, grid_size, num_baselines);
			num_blocks = computeCeil(static_cast<float>(grid_size * grid_size)/num_threads);
			combineToComplex<<<num_blocks,num_threads,0,stream1>>>(w_grid_stack_real, w_grid_stack_imag, w_grid_stack, grid_size);
			num_threads = 32;
			numThreads.x = num_threads;
			numThreads.y = num_threads;
			numBlocks.x = computeCeil(static_cast<float>(grid_size)/num_threads);
			numBlocks.y = computeCeil(static_cast<float>(grid_size)/num_threads);
			ifftShift<<<numBlocks,numThreads,0,stream1>>>(w_grid_stack, w_grid_stack_shifted, grid_size, grid_size);
			cufftExecC2C(plan, w_grid_stack_shifted, w_grid_stack_shifted, CUFFT_INVERSE);
			numBlocks.x = computeCeil(static_cast<float>(image_size)/num_threads);
			numBlocks.y = computeCeil(static_cast<float>(image_size)/num_threads);
			accumulation<<<numBlocks,numThreads,0,stream1>>>(dirty_pre, w_grid_stack_shifted, image_size, grid_size);
			scaling<<<numBlocks,numThreads,0,stream1>>>(dirty_pre, conv_corr_kernel, image_size, conv_corr_norm_factor);
			coordschange<<<numBlocks,numThreads,0,stream2>>>(output_index, V_in, image_size);
			p2p<<<numBlocks,numThreads,0,stream2>>>(output_index, V_in, cell_size, image_size);
			
			cudaEventRecord(eventstream[ind-1],stream2);
			cudaStreamWaitEvent(stream1,eventstream[ind-1],0);
			if (ind == 1) {
				finalinterp<<<numBlocks,numThreads,0,stream1>>>(output_index, dirty_pre, dirty1, image_size);
				num_threads = 1024;
				num_blocks = region_num*region_num;
				shared_mem_size = num_threads * sizeof(float);
				max_large<<<num_blocks,num_threads,shared_mem_size,stream1>>>(dirty1, max_tmp, region_num, region_size, image_size);
				num_blocks = 1;
				bid_ind = ind - 1;
				max_small<<<num_blocks,num_threads,shared_mem_size,stream1>>>(max_tmp, maxall, image_size, bid_ind);
			}
			if (ind == 2) {
				finalinterp<<<numBlocks,numThreads,0,stream1>>>(output_index, dirty_pre, dirty2, image_size);
				num_threads = 1024;
				num_blocks = region_num*region_num;
				shared_mem_size = num_threads * sizeof(float);
				max_large<<<num_blocks,num_threads,shared_mem_size,stream1>>>(dirty2, max_tmp, region_num, region_size, image_size);
				num_blocks = 1;
				bid_ind = ind - 1;
				max_small<<<num_blocks,num_threads,shared_mem_size,stream1>>>(max_tmp, maxall, image_size, bid_ind);
			}
			
			cudaEventRecord(events_kernel[ind],stream1);
			
			cudaStreamWaitEvent(stream_pcie,events_kernel[ind],0);
			cudaMemcpyAsync(Vis_real, Vis_realtmp, num_baselines * 1 * sizeof(float), cudaMemcpyDeviceToDevice, stream_pcie);
			cudaMemcpyAsync(Vis_imag, Vis_imagtmp, num_baselines * 1 * sizeof(float), cudaMemcpyDeviceToDevice, stream_pcie);
			cudaMemcpyAsync(B_in, B_intmp, num_baselines * 2 * sizeof(float), cudaMemcpyDeviceToDevice, stream_pcie);
			cudaMemcpyAsync(V_in, V_intmp, 3 * 3 * sizeof(float), cudaMemcpyDeviceToDevice, stream_pcie);
			cudaEventRecord(events[ind],stream_pcie);
		}
	}
	
	cudaStreamWaitEvent(stream1,events[2],0);
	cudaStreamWaitEvent(stream2,events[2],0);
	
	cudaMemsetAsync(dirty_pre, 0, image_size * image_size * sizeof(float),stream1);
	cudaMemsetAsync(conv_corr_kernel, 0, (image_size/2+1) * sizeof(float),stream2);
	cudaMemsetAsync(w_grid_stack_real, 0, grid_size * grid_size * sizeof(float),stream1);
	cudaMemsetAsync(w_grid_stack_imag, 0, grid_size * grid_size * sizeof(float),stream1);
	cudaMemsetAsync(output_index, 0, image_size * image_size * 2 * sizeof(float),stream2);
			
	num_threads = 1024;
	num_blocks = computeCeil(static_cast<float>(image_size/2+1)/num_threads);
	convolveKernel<<<num_blocks,num_threads,0,stream2>>>(conv_corr_kernel, image_size, grid_size, conv_corr_norm_factor);
	num_blocks = computeCeil(static_cast<float>(num_baselines)/num_threads);
	computeVisWeighted<<<num_blocks,num_threads,0,stream1>>>(Vis_real,Vis_imag,num_baselines,V_in);
	gridding<<<num_blocks,num_threads,0,stream1>>>(B_in, w_grid_stack_real, w_grid_stack_imag, Vis_real, Vis_imag, freq_hz, uv_scale, grid_size, num_baselines);
	num_blocks = computeCeil(static_cast<float>(grid_size * grid_size)/num_threads);
	combineToComplex<<<num_blocks,num_threads,0,stream1>>>(w_grid_stack_real, w_grid_stack_imag, w_grid_stack, grid_size);
	num_threads = 32;
	numThreads.x = num_threads;
	numThreads.y = num_threads;
	numBlocks.x = computeCeil(static_cast<float>(grid_size)/num_threads);
	numBlocks.y = computeCeil(static_cast<float>(grid_size)/num_threads);
	ifftShift<<<numBlocks,numThreads,0,stream1>>>(w_grid_stack, w_grid_stack_shifted, grid_size, grid_size);
	cufftExecC2C(plan, w_grid_stack_shifted, w_grid_stack_shifted, CUFFT_INVERSE);
	numBlocks.x = computeCeil(static_cast<float>(image_size)/num_threads);
	numBlocks.y = computeCeil(static_cast<float>(image_size)/num_threads);
	accumulation<<<numBlocks,numThreads,0,stream1>>>(dirty_pre, w_grid_stack_shifted, image_size, grid_size);
	scaling<<<numBlocks,numThreads,0,stream1>>>(dirty_pre, conv_corr_kernel, image_size, conv_corr_norm_factor);
	coordschange<<<numBlocks,numThreads,0,stream2>>>(output_index, V_in, image_size);
	p2p<<<numBlocks,numThreads,0,stream2>>>(output_index, V_in, cell_size, image_size);
		
	cudaEventRecord(eventstream[2],stream2);
	cudaStreamWaitEvent(stream1,eventstream[2],0);
	finalinterp<<<numBlocks,numThreads,0,stream1>>>(output_index, dirty_pre, dirty3, image_size);
	num_threads = 1024;
	num_blocks = region_num*region_num;
	shared_mem_size = num_threads * sizeof(float);
	max_large<<<num_blocks,num_threads,shared_mem_size,stream1>>>(dirty3, max_tmp, region_num, region_size, image_size);
	num_blocks = 1;
	bid_ind = 2;
	max_small<<<num_blocks,num_threads,shared_mem_size,stream1>>>(max_tmp, maxall, image_size, bid_ind);
	
	cudaStreamSynchronize(stream1);
	
	/* ****************************************************** */
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout << "Time elapsed: " << milliseconds << " ms" << std::endl;
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
	for (int i = 0; i < 3; i++){
		cudaEventDestroy(eventstream[i]);
		cudaEventDestroy(events[i]);
		cudaEventDestroy(events_kernel[i]);
	}
	
	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream2);
	cudaStreamDestroy(stream_pcie);
    
	cudaFree(dirty_pre);
	cudaFree(conv_corr_kernel);
	cudaFree(w_grid_stack_real);
	cudaFree(w_grid_stack_imag);
	cudaFree(w_grid_stack);
	cudaFree(w_grid_stack_shifted);
	cudaFree(output_index);
	cudaFree(pixel_ind);
	cudaFree(max_tmp);
	cudaFree(Vis_realtmp);
	cudaFree(Vis_imagtmp);
	cudaFree(B_intmp);
	cudaFree(V_intmp);
	cudaFree(Vis_real);
	cudaFree(Vis_imag);
	cudaFree(B_in);
	cudaFree(V_in);
	
	cudaFreeHost(pinned_Vis_real);
	cudaFreeHost(pinned_Vis_imag);
	cudaFreeHost(pinned_B_in);
	cudaFreeHost(pinned_V_in);
	
	// FI Trigger
	float* d_data_1;
	float* d_data_2;
	float* diff_out;
	float* result_data;
	float C = 1e-6;
    
	size_t unit_num = image_size/unit_size;
    
	cudaMalloc((void**)&d_data_1, image_size * image_size * sizeof(float));
	cudaMalloc((void**)&d_data_2,  image_size * image_size * sizeof(float));
	cudaMalloc((void**)&diff_out,  image_size * image_size * sizeof(float));
	cudaMalloc((void**)&result_data, unit_num * unit_num * sizeof(float));
    
	cudaMemset(d_data_1, 0, image_size * image_size * sizeof(float));
	cudaMemset(d_data_2, 0, image_size * image_size * sizeof(float));
	cudaMemset(diff_out, 0, image_size * image_size * sizeof(float));
	cudaMemset(result_data, 0, unit_num * unit_num * sizeof(float));

	cudaEvent_t start1, stop1;
	cudaEventCreate(&start1);
	cudaEventCreate(&stop1);

	cudaEventRecord(start1);
	/* ****************************************************** */
	size_t size = image_size * image_size;
	num_threads = 1024;
	num_blocks = computeCeil(static_cast<float>(size)/num_threads);
	subtraction<<<num_blocks,num_threads>>>(dirty1, dirty2, d_data_1, size);
	subtraction<<<num_blocks,num_threads>>>(dirty2, dirty3, d_data_2, size);
	setNonPositiveToC<<<num_blocks,num_threads>>>(dirty2, size, C); // Be careful about this, because when there are a lot of trigger, this may cause trouble (because the image dirty2 has been changed by this).
	subtraction<<<num_blocks,num_threads>>>(d_data_1, d_data_2, diff_out, size);
	num_blocks = unit_num*unit_num;
	shared_mem_size = 3 * num_threads * sizeof(float);
	tlisi<<<num_blocks,num_threads,shared_mem_size>>>(diff_out, dirty2, result_data, unit_size, image_size, unit_num, maxall);
	/* ****************************************************** */
	cudaEventRecord(stop1);
	cudaEventSynchronize(stop1);
	float milliseconds1 = 0;
	cudaEventElapsedTime(&milliseconds1, start1, stop1);
	std::cout << "Time elapsed: " << milliseconds1 << " ms" << std::endl;

	cudaEventDestroy(start1);
	cudaEventDestroy(stop1);
	
	cudaMemcpy(result_array, result_data, unit_num * unit_num * sizeof(float), cudaMemcpyDeviceToHost);
	
	cudaFree(d_data_1);
	cudaFree(d_data_2);
	cudaFree(diff_out);
	cudaFree(result_data);
	cudaFree(dirty1);
	cudaFree(dirty2);
	cudaFree(dirty3);
	
	return 0;
}
	
