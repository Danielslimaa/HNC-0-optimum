#pragma once
#include <math.h>
#include <cmath>
#include <cuda_runtime.h>

#define MAX_BLKSZ 1024
#define WARPSZ 32
#define BLOCK_SIZE 32

extern int Blocks_N;
extern int ThreadsPerBlock_N;

__device__ __managed__ double U;
__device__ __managed__ double rho;
__device__ __managed__ double sigma;
__device__ __managed__ double dt;
__device__ __managed__ double TildeRc;
__device__ __managed__ double dr;
__device__ __managed__ double dk;
__device__ __managed__ double precision;
__device__ __managed__ long int N;

int Blocks_N, ThreadsPerBlock_N, aux_total, total;
int M = 100000; 

//__device__ __managed__ double dU;
//__device__ __managed__ double memoria_dU;

#if __CUDA_ARCH__ < 600
__device__ double atomicAdd_double(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

__device__ double Warp_sum(double var)
{
    unsigned mask = 0xffffffff;

    for (int diff = warpSize / 2; diff > 0; diff = diff / 2)
        var += __shfl_down_sync(mask, var, diff, 32);
    return var;
} /* Warp_sum */

__device__ double Shared_mem_sum(double shared_vals[])
{
    int my_lane = threadIdx.x % warpSize;

    for (int diff = warpSize / 2; diff > 0; diff = diff / 2)
    {
        /* Make sure 0 <= source < warpSize  */
        int source = (my_lane + diff) % warpSize;
        shared_vals[my_lane] += shared_vals[source];
    }
    return shared_vals[my_lane];
}

__global__ void part_0_Dev_trap(
    const double g[]     /* in       */,
    const double S[]     /* in       */,
    const double potential[]     /* in       */,
    double* trap_p  /* in/out   */)
{
    __shared__ double thread_calcs[MAX_BLKSZ];
    __shared__ double warp_sum_arr[WARPSZ];
    int my_i = blockDim.x * blockIdx.x + threadIdx.x;
    int my_warp = threadIdx.x / warpSize;
    int my_lane = threadIdx.x % warpSize;
    double *shared_vals = thread_calcs + my_warp * warpSize;
    double blk_result = 0.0;
    
    shared_vals[my_lane] = 0.0;
    if (my_i == 0 || my_i == N - 2)
    {
        shared_vals[my_lane] = 0.5 * (3.14159265359 * rho) * dr * dr * g[my_i] * potential[my_i] * (double)(my_i);
        shared_vals[my_lane] += 0.5 * (-0.01989436788 / rho) * dk * dk * dk * dk * pow(((double)my_i) * (S[my_i]-1.0), 3) / S[my_i];
        shared_vals[my_lane] += 0.5 * (-0.78539816339 *rho) * ( ((double)(my_i)) * (3.0*(g[my_i+1]) + (g[my_i-1])\
         - (g[my_i+1]*g[my_i+1]/g[my_i]) - g[my_i] * 3.0) + ((g[my_i+1]) - g[my_i] * 1.0));
    }
    if (0 < my_i && my_i < N - 2)
    {
        shared_vals[my_lane] = (3.14159265359 * rho) * dr * dr * g[my_i] * potential[my_i] * (double)(my_i);
        shared_vals[my_lane] += (-0.01989436788 / rho) * dk * dk * dk * dk * pow(((double)my_i) * (S[my_i]-1.0), 3) / S[my_i];
        shared_vals[my_lane] += (-0.78539816339 * rho) * ( ((double)(my_i)) * (3.0*(g[my_i+1]) + (g[my_i-1])\
         - (g[my_i+1]*g[my_i+1]/g[my_i]) - g[my_i] * 3.0) + ((g[my_i+1]) - g[my_i] * 1.0));
    }

    double my_result = Shared_mem_sum(shared_vals);
    if (my_lane == 0)
    {
        warp_sum_arr[my_warp] = my_result;
    }
    __syncthreads();

    if (my_warp == 0)
    {
        if (threadIdx.x >= blockDim.x / warpSize)
        {
            warp_sum_arr[threadIdx.x] = 0.0;
        }
        blk_result = Shared_mem_sum(warp_sum_arr);
    }

    if (threadIdx.x == 0)
    {
        atomicAdd_double(trap_p, blk_result);
    }
} /* Dev_trap */

void energy_CUDA_wrapper(
                        const double g[],
                        const double S[],
                        const double potential[],
                        double *energia_p
)
{
    energia_p[0] = 0.0;
	part_0_Dev_trap<<<Blocks_N, ThreadsPerBlock_N>>>(g, S, potential, energia_p);
    cudaDeviceSynchronize();
}

__global__ void auto_energia_Dev_trap(
    double *d_potential     /* in       */,
    long int n  /* in       */,
    double *trap_p  /* in/out   */)
{
    __shared__ double thread_calcs[MAX_BLKSZ];
    __shared__ double warp_sum_arr[WARPSZ];
    int my_i = blockDim.x * blockIdx.x + threadIdx.x;
    int my_warp = threadIdx.x / warpSize;
    int my_lane = threadIdx.x % warpSize;
    double *shared_vals = thread_calcs + my_warp * warpSize;
    double blk_result = 0.0;

    shared_vals[my_lane] = 0.0;
    if (my_i == 0 || my_i == n - 1)
    {
        shared_vals[my_lane] = 0.5 * d_potential[my_i] * (double)(my_i);
    }
    else
    {
        shared_vals[my_lane] = d_potential[my_i] * (double)(my_i);
    }

    double my_result = Shared_mem_sum(shared_vals);
    if (my_lane == 0)
    {
        warp_sum_arr[my_warp] = my_result;
    }
    __syncthreads();

    if (my_warp == 0)
    {
        if (threadIdx.x >= blockDim.x / warpSize)
        {
            warp_sum_arr[threadIdx.x] = 0.0;
        }
        blk_result = Shared_mem_sum(warp_sum_arr);
    }

    if (threadIdx.x == 0)
    {
        atomicAdd_double(trap_p, 0.5 * (rho) * dr * dr * blk_result);
    }
} /* Dev_trap */

__global__ void entropia_Dev_trap(
    double *d_g     /* in       */,
    long int n  /* in       */,
    double *trap_p  /* in/out   */)
{
    __shared__ double thread_calcs[MAX_BLKSZ];
    __shared__ double warp_sum_arr[WARPSZ];
    int my_i = blockDim.x * blockIdx.x + threadIdx.x;
    int my_warp = threadIdx.x / warpSize;
    int my_lane = threadIdx.x % warpSize;
    double *shared_vals = thread_calcs + my_warp * warpSize;
    double blk_result = 0.0;

    shared_vals[my_lane] = 0.0;
    if (my_i == 0 || my_i == n - 1)
    {
        shared_vals[my_lane] = 0.5 * ( d_g[my_i] * log( d_g[my_i] ) - d_g[my_i] + 1.0 ) * (double)(my_i);
    }
    else
    {
        shared_vals[my_lane] = ( d_g[my_i] * log( d_g[my_i] ) - d_g[my_i] + 1.0 ) * (double)(my_i);
    }

    double my_result = Shared_mem_sum(shared_vals);
    if (my_lane == 0)
    {
        warp_sum_arr[my_warp] = my_result;
    }
    __syncthreads();

    if (my_warp == 0)
    {
        if (threadIdx.x >= blockDim.x / warpSize)
        {
            warp_sum_arr[threadIdx.x] = 0.0;
        }
        blk_result = Shared_mem_sum(warp_sum_arr);
    }

    if (threadIdx.x == 0)
    {
        atomicAdd_double(trap_p, -(2.0 * M_PI * rho) * dr * dr * blk_result);
    }
} /* Dev_trap */

void entropia_CUDA_wrapper(
                        double* &d_g,
                        const int n,
                        double* &entropia
)
{
    *entropia = 0.0;
    entropia_Dev_trap<<<Blocks_N, ThreadsPerBlock_N>>>(d_g, n, entropia);
    cudaDeviceSynchronize();
}

void auto_energia_wrapper(
                        double* &d_potential,
                        const int N,
                        double* &hartree
)
{
    *hartree = 0.0;
    entropia_Dev_trap<<<Blocks_N, ThreadsPerBlock_N>>>(d_potential, N, hartree);
    cudaDeviceSynchronize();
}

void printer_vector(double *vetor, const char *name, double dx)
{
	std::ofstream myfile;
	myfile.open(name);
	for (int i = 0; i < N; ++i)
	{
		myfile << (double)i * dx << "," << vetor[i] << "\n";
	}
	myfile.close();
	return;
}

__host__ void compute_potential(double *&potential)
{
	/* LPG8 no topo do diagrama de fases  
	sigma = 0.975;
	double d0 = 25.;
	double d1 = -184.36912759277425;
	double d2 = 780.7479127964228;
	double d3 = -1704.7876063560543;
	double d4 = 1969.9440901262803;
	double d5 = -1258.7937954278757;
	double d6 = 444.60543492282784;
	double d7 = -80.81042877960013;
	double d8 = 5.876194077794321;
	*/
	
	/* LPG8 no topo do diagrama de fases*/
	sigma = 0.8;
	double d0 = 25.;
	double d1 = 191.5260825731478;
	double d2 = -854.2206943713246;
	double d3 = 1232.3821939383304;
	double d4 = -841.1360147950575;
	double d5 = 284.9669873459703;
	double d6 = -40.79348857357674;
	double d7 = 0.0025935420530466056;
	double d8 = 0.37585946115394886;
	

  for (int i = 0; i < N; ++i)
  {
    double k = ((double)i) * dk;
    double aux = d0+d1*pow(k,2)+d2*pow(k,4)+d3*pow(k,6)+d4*pow(k,8)+d5*pow(k,10)+d6*pow(k,12)+d7*pow(k,14)+d8*pow(k,16);
    potential[i] = U * exp(-pow(sigma * k, 2)) * aux; 
  }

  //printer_vector(potential, "potential_LPGE8", dk);

	double C0 = 65536. * pow(sigma, 32) * d0 + 65536. * pow(sigma, 30) * d1;
	C0 += 131072. * pow(sigma, 28) * d2 + 393216. * pow(sigma, 26) * d3;
	C0 += 1572864. * pow(sigma, 24) * d4 + 7864320. * pow(sigma, 22) * d5;
	C0 += 47185920. * pow(sigma, 20) * d6 + 330301440. * pow(sigma, 18) * d7;
	C0 += 2642411520. * pow(sigma, 16) * d8;

	double C1 = 16384. * pow(sigma, 28) * d1 + 65536. * pow(sigma, 26) * d2;
	C1 += 294912. * pow(sigma, 24) * d3 + 1572864. * pow(sigma, 22) * d4;
	C1 += 9830400. * pow(sigma, 20) * d5 + 70778880. * pow(sigma, 18) * d6;
	C1 += 578027520. * pow(sigma, 16) * d7 + 5284823040. * pow(sigma, 14) * d8;
	C1 *= -1;

	double C2 = 4096. * pow(sigma, 24) * d2 + 36864. * pow(sigma, 22) * d3;
	C2 += 294912. * pow(sigma, 20) * d4 + 2457600. * pow(sigma, 18) * d5;
	C2 += 22118400. * pow(sigma, 16) * d6 + 216760320. * pow(sigma, 14) * d7;
	C2 += 2312110080. * pow(sigma, 12) * d8;

	double C3 = 1024. * pow(sigma, 20) * d3 + 16384 * pow(sigma, 18) * d4;
	C3 += 204800. * pow(sigma, 16) * d5 + 2457600. * pow(sigma, 14) * d6;
	C3 += 30105600. * pow(sigma, 12) * d7 + 385351680. * pow(sigma, 10) * d8;
	C3 *= -1;

	double C4 = 256. * pow(sigma, 16) * d4 + 6400. * pow(sigma, 14) * d5;
	C4 += 115200. * pow(sigma, 12) * d6 + 1881600. * pow(sigma, 10) * d7;
	C4 += 30105600. * pow(sigma, 8) * d8;

	double C5 = 64. * pow(sigma, 12) * d5 + 2304. * pow(sigma, 10) * d6;
	C5 += 56448. * pow(sigma, 8) * d7 + 1204224. * pow(sigma, 6) * d8;
	C5 *= -1;

	double C6 = 16. * pow(sigma, 8) * d6 + 784. * pow(sigma, 6) * d7;
	C6 += 25088. * pow(sigma, 4) * d8;

	double C7 = 4. * pow(sigma, 4) * d7 + 256. * pow(sigma, 2) * d8;
	C7 *= -1;

	double C8 = d8;

	C0 /= (262144. * M_PI * pow(sigma, 34));
	C1 /= (262144. * M_PI * pow(sigma, 34));
	C2 /= (262144. * M_PI * pow(sigma, 34));
	C3 /= (262144. * M_PI * pow(sigma, 34));
	C4 /= (262144. * M_PI * pow(sigma, 34));
	C5 /= (262144. * M_PI * pow(sigma, 34));
	C6 /= (262144. * M_PI * pow(sigma, 34));
	C7 /= (262144. * M_PI * pow(sigma, 34));
	C8 /= (262144. * M_PI * pow(sigma, 34));

	//printf("\n C0 = %.6f, C1 = %.6f, C2 = %.6f, C3 = %.6f, C4 = %.6f, C5 = %.6f, C6 = %.6f, C7 = %.6f, C8 = %.6f\n", C0, C1, C2, C3, C4, C5, C6, C7, C8);
	// printf("U = %.6f\n", U);

	for (int i = 0; i < N; ++i)
	{
		double r = ((double)i) * dr;
		double aux = C0 + C1 * pow(r, 2) + C2 * pow(r, 4) + C3 * pow(r, 6) + C4 * pow(r, 8) + C5 * pow(r, 10) + C6 * pow(r, 12) + C7 * pow(r, 14) + C8 * pow(r, 16);
		potential[i] =  U * exp(-pow(r / (2. * sigma), 2)) * aux; //U / (1.0 + pow(r, 6));//
	}
	
}

__global__ void BesselJ0Table(double *d_j0table)
{
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int stride_x = blockDim.x * gridDim.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	int stride_y = blockDim.y * gridDim.y;
	int i, j;
	for (i = index_x; i < N; i += stride_x)
	{
		for (j = index_y; j < N; j += stride_y)
		{
			d_j0table[i * N + j] = j0f(double(i) * dr * double(j) * dk);
		}
	}
}

__global__ void g_from_S(double *d_g, double *d_S, double *d_j0table)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	double aux;
	if (row < N)
	{
		aux = 0.0;
		for (int i = 0; i < N; i++)
		{
			aux += double(i) * j0f(double(row) * dr * double(i) * dk) * (d_S[i] - 1.0);
		}
		d_g[row] = 1.0 + aux * dk * dk / (2.0 * M_PI * rho);
	}
}

__global__ void S_from_g(double *d_g, double *d_S, double *d_j0table)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	double aux;
	if (col < N)
	{
		aux = 0.0;
		for (int i = 0; i < N; i++)
		{
			aux += double(i) * j0f(double(i) * dr * double(col) * dk) * (d_g[i] - 1.0);
		}
		d_S[col] = 1.0 + aux * dr * dr * 2.0 * M_PI * rho;
	}
}

__global__ void calculate_first_term(double *d_g, double *d_first_term, double *potential)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N)
	{
		d_first_term[i] = d_g[i] * potential[i];
	}
}

__global__ void calculate_second_term(double *d_g, double *d_second_term)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	double aux;
	if (i < N - 7)
	{
		aux = ((-147.0 * d_g[i] + 360.0 * d_g[i + 1] - 450.0 * d_g[i + 2] + 400.0 * d_g[i + 3] - 225.0 * d_g[i + 4] + 72.0 * d_g[i + 5] - 10.0 * d_g[i + 6]) / (60.0 * dr));
		// aux = (d_g[i + 1] - d_g[i]) / dr;
		d_second_term[i] = aux * aux / (4.0 * d_g[i]);
	}
	if (i > N - 6)
	{
		aux = ((d_g[i] - d_g[i - 1]) / dr);
		d_second_term[i] = aux * aux / (4.0 * d_g[i]); // g(r) is even, then its derivative is odd
	}
}

__global__ void calculate_second_term2(double *d_g, double *d_second_term)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	double aux1, aux2;

	if (i < N - 7)
	{
		aux1 = ((-147.0 * d_g[i] + 360.0 * d_g[i + 1] - 450.0 * d_g[i + 2] + 400.0 * d_g[i + 3] - 225.0 * d_g[i + 4] + 72.0 * d_g[i + 5] - 10.0 * d_g[i + 6]) / (60.0));
		aux2 = (812.0 * d_g[i + 0] - 3132.0 * d_g[i + 1] + 5265.0 * d_g[i + 2] - 5080.0 * d_g[i + 3] + 2970.0 * d_g[i + 4] - 972.0 * d_g[i + 5] + 137.0 * d_g[i + 6]) / (180);
		d_second_term[i] = -0.25 * ((-((double)i) / d_g[i]) * aux1 * aux1 + ((double)i) * aux2) / (dr * dr);
	}
	if (i < N - 2)
	{
		d_second_term[i] += -0.25 * (((double)(i)) * (3.0 * (d_g[i + 1]) + (d_g[i - 1]) - (d_g[i + 1] * d_g[i + 1] / d_g[i]) - d_g[i] * 3.0) + ((d_g[i + 1]) - d_g[i] * 1.0)) / (dr * dr);
	}
	if (i == N - 1)
	{
		d_second_term[i] = d_second_term[N - 2];
	}
}

__global__ void calculate_third_term(double *d_S, double *d_j0table,
									 double *d_third_term)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	double aux;
	int j;
	double epsilon = 1e-15;
	aux = 0.0;

	for (j = 1; j < N; ++j)
	{
		aux += -0.5 * double(j) * double(j) * double(j) * j0f(double(j) * dr * double(col) * dk) * (2.0 * d_S[j] + 1.0) * pow(1.0 - (1.0 / (epsilon + d_S[j])), 2);
	}
	d_third_term[col] = aux * dk * dk * dk * dk / (2.0 * M_PI * rho);
}

__global__ void calculate_V_ph_k(double *d_first_term, double *d_second_term,
								 double *d_third_term, double *d_kernel, double *d_V_ph_k)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	double aux;
	long int j;
	if (row < N)
	{
		aux = 0.0;
		for (j = 0; j < N; ++j)
		{
			aux += double(j) * j0f(double(row) * dr * double(j) * dk) * (d_first_term[j] + 2.0 * d_second_term[j] + d_third_term[j]);
		}
		d_V_ph_k[row] = aux * dr * dr * 2.0 * M_PI * rho;
	}
}

__global__ void update_S(double *d_V_ph_k, double *d_S, int *checker)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	// double epsilon = 1e-15;
	if (checker[0] == 0)
	{
		if (i < N)
		{
			d_S[i] = dt * (double(i) * dk) / sqrt(fabs(pow(double(i) * dk, 2) + 2.0 * d_V_ph_k[i])) + (1.0 - dt) * d_S[i];
		}
	}
	if (checker[0] == 1)
	{
		if (i < N)
		{
			d_S[i] = dt * (double(i) * dk) / sqrt(pow(double(i) * dk, 2) + 2.0 * d_V_ph_k[i]) + (1.0 - dt) * d_S[i];
		}
	}
}

__global__ void update_S2(const double *d_V_ph_k, double *d_S, int *checker)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	// double e = 1e-9;
	double new_S;

	if (checker[0] == 0)
	{
		if (i > 0 && i < N)
		{
			new_S = (double(i) * dk) / sqrt(fabs(pow(double(i) * dk, 2) + 4.0 * d_V_ph_k[i]));
			// d_S[i] = d_S[i] * (1.0 - dt) + dt * new_S;
			d_S[i] = new_S;
		}
		if (i == 0)
		{
			d_S[i] = d_S[i + 1];
		}
	}
	if (checker[0] == 1)
	{
		if (i > 0 && i < N)
		{
			new_S = (double(i) * dk) / sqrt((pow(double(i) * dk, 2) + 4.0 * d_V_ph_k[i]));
			// d_S[i] = d_S[i] * (1.0 - dt) + dt * new_S;
			d_S[i] = new_S;
		}
		if (i == 0)
		{
			d_S[i] = d_S[i + 1];
		}
	}
}

__global__ void calculate_V_ph_k2(
	const double *__restrict__ dx1,
	const double *__restrict__ dx2,
	const double *__restrict__ dx3,
	double *__restrict__ d_V_ph_k,
	const int nRows,
	const int nCols)
{
	const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

	double x_shfl_src, x_shfl_dest;

	double y_val = 0.0;

#pragma unroll
	for (unsigned int m = 0; m < ((nCols + BLOCK_SIZE - 1) / BLOCK_SIZE); ++m)
	{
		if ((m * BLOCK_SIZE + threadIdx.x) < nCols)
		{
			x_shfl_src = dx1[threadIdx.x + m * BLOCK_SIZE] + 1.0 * dx2[threadIdx.x + m * BLOCK_SIZE] + dx3[threadIdx.x + m * BLOCK_SIZE] ;
			x_shfl_src *= (double)(threadIdx.x + m * BLOCK_SIZE);
		}
		else
		{
			x_shfl_src = 0.0;
		}
		__syncthreads();
		// #pragma unroll
		for (int e = 0; e < 32; ++e)
		{
			// --- Column-major ordering - faster
			x_shfl_dest = __shfl_sync(0xffffffff, x_shfl_src, e);
			// y_val += d_j0table[tid * nCols + (e + BLOCK_SIZE * m)] * x_shfl_dest;
			y_val += j0f(double(tid) * dr * double(e + BLOCK_SIZE * m) * dk) * x_shfl_dest;
			// --- Row-major ordering - slower
			// y_val += d_V_ph_k[tid * nCols + (e + BLOCK_SIZE * m)] * x_shared[e];
		}
		__syncthreads();
	}
	if (tid < nRows)
	{
		d_V_ph_k[tid] = y_val * dr * dr * 2.0 * M_PI * rho;
	}
}

__global__ void calculate_third_term2(
	const double *__restrict__ g,
	const double *__restrict__ S,
	double *__restrict__ third_term,
	const int nRows,
	const int nCols)
{
	const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

	double x_shfl_src, x_shfl_dest;

	double y_val = 0.0;

#pragma unroll
	for (unsigned int m = 0; m < ((nCols + BLOCK_SIZE - 1) / BLOCK_SIZE); ++m)
	{
		if ((m * BLOCK_SIZE + threadIdx.x) < nCols)
		{
			x_shfl_src = (2.0 * S[threadIdx.x + m * BLOCK_SIZE] + 1.0) * pow(1.0 - 1.0 / S[threadIdx.x + m * BLOCK_SIZE], 2);
			x_shfl_src *= -0.25 * pow((double)(threadIdx.x + m * BLOCK_SIZE), 3);
		}
		else
		{
			x_shfl_src = 0.0;
		}
		__syncthreads();

		//        #pragma unroll
		for (int e = 0; e < 32; ++e)
		{
			// --- Column-major ordering - faster
			x_shfl_dest = __shfl_sync(0xffffffff, x_shfl_src, e);
			// y_val += d_j0table[tid * nCols + (e + BLOCK_SIZE * m)] * x_shfl_dest;
			y_val += j0f(double(tid) * dr * double(e + BLOCK_SIZE * m) * dk) * x_shfl_dest;
			// --- Row-major ordering - slower
			// y_val += d_V_ph_k[tid * nCols + (e + BLOCK_SIZE * m)] * x_shared[e];
		}
		__syncthreads();
	}
	if (tid < nRows)
	{
		third_term[tid] = y_val * (g[tid] - 1.0) * dk * dk * dk * dk * (1.0 / (2.0 * M_PI * rho));
	}
}

__global__ void calculate_third_term_final(
	const double *__restrict__ S,
	const double *__restrict__ d_j0table,
	double *__restrict__ third_term,
	const int nRows,
	const int nCols)
{
	const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

	double x_shfl_src, x_shfl_dest;

	double y_val = 0.0;

#pragma unroll
	for (unsigned int m = 0; m < ((nCols + BLOCK_SIZE - 1) / BLOCK_SIZE); ++m)
	{
		if ((m * BLOCK_SIZE + threadIdx.x) < nCols)
		{
			x_shfl_src = (2.0 * S[threadIdx.x + m * BLOCK_SIZE] + 1.0) * pow(1.0 - 1.0 / (S[threadIdx.x + m * BLOCK_SIZE] + 1e-9), 2);
			x_shfl_src *= -0.5 * pow((double)(threadIdx.x + m * BLOCK_SIZE), 3);
		}
		else
		{
			x_shfl_src = 0.0;
		}
		__syncthreads();

		//        #pragma unroll
		for (int e = 0; e < 32; ++e)
		{
			// --- Column-major ordering - faster
			x_shfl_dest = __shfl_sync(0xffffffff, x_shfl_src, e);
			// y_val += d_j0table[tid * nCols + (e + BLOCK_SIZE * m)] * x_shfl_dest;
			y_val += j0f(double(tid) * dr * double(e + BLOCK_SIZE * m) * dk) * x_shfl_dest;
			// --- Row-major ordering - slower
			// y_val += d_V_ph_k[tid * nCols + (e + BLOCK_SIZE * m)] * x_shared[e];
		}
		__syncthreads();
	}
	if (tid < nRows)
	{
		third_term[tid] = y_val * dk * dk * dk * dk / (2.0 * M_PI * rho);
	}
}

__global__ void g_from_S2(
	double *__restrict__ g,
	const double *__restrict__ S,
	const double *__restrict__ d_j0table,
	const int nRows,
	const int nCols,
	double *sum_diff_g)
{
	const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

	double x_shfl_src, x_shfl_dest;

	double y_val = 0.0;

#pragma unroll
	for (unsigned int m = 0; m < ((nCols + BLOCK_SIZE - 1) / BLOCK_SIZE); ++m)
	{
		if ((m * BLOCK_SIZE + threadIdx.x) < nCols)
		{
			x_shfl_src = S[threadIdx.x + m * BLOCK_SIZE] - 1.0;
			x_shfl_src *= (double)(threadIdx.x + m * BLOCK_SIZE);
		}
		else
		{
			x_shfl_src = 0.0;
		}
		__syncthreads();

		//        #pragma unroll
		for (int e = 0; e < 32; ++e)
		{
			// --- Column-major ordering - faster
			x_shfl_dest = __shfl_sync(0xffffffff, x_shfl_src, e);
			y_val += j0f(double(tid) * dr * double(e + BLOCK_SIZE * m) * dk) * x_shfl_dest;
			// --- Row-major ordering - slower
			// y_val += d_V_ph_k[tid * nCols + (e + BLOCK_SIZE * m)] * x_shared[e];
		}

		__syncthreads();
	}

	if (tid < nRows)
	{
		sum_diff_g[0] += fabs(1.0 + y_val * dk * dk / (2.0 * M_PI * rho) - g[tid]);
		g[tid] = 1.0 + y_val * dk * dk / (2.0 * M_PI * rho);
	}
}

__global__ void S_from_g2(
	const double *__restrict__ g,
	double *__restrict__ S)
{
	const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

	double x_shfl_src, x_shfl_dest;

	double y_val = 0.0;

	#pragma unroll
	for (unsigned int m = 0; m < ((N + BLOCK_SIZE - 1) / BLOCK_SIZE); ++m)
	{
		if ((m * BLOCK_SIZE + threadIdx.x) < N)
		{
			x_shfl_src = g[threadIdx.x + m * BLOCK_SIZE] - 1.0;
			x_shfl_src *= (double)(threadIdx.x + m * BLOCK_SIZE);
		}
		else
		{
			x_shfl_src = 0.0;
		}
		__syncthreads();

		//        #pragma unroll
		for (int e = 0; e < 32; ++e)
		{
			// --- Column-major ordering - faster
			x_shfl_dest = __shfl_sync(0xffffffff, x_shfl_src, e);
			// y_val += d_j0table[tid * nCols + (e + BLOCK_SIZE * m)] * x_shfl_dest;
			y_val += j0f(double(tid) * dr * double(e + BLOCK_SIZE * m) * dk) * x_shfl_dest;
			// --- Row-major ordering - slower
			// y_val += d_V_ph_k[tid * nCols + (e + BLOCK_SIZE * m)] * x_shared[e];
		}

		__syncthreads();
	}

	if (tid < N)
	{
		S[tid] = 1.0 + y_val * dr * dr * 2.0 * M_PI * rho;
	}
}

__global__ void maxonroton(
	double *__restrict__ S,
	double *__restrict__ maxon,
	double *__restrict__ maxon_position,
	double *__restrict__ roton,
	double *__restrict__ roton_position)
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	double position_m3, position_m2, position_m1, position_0, position_p1, position_p2, position_p3;
	double aux, aux2;
	double aux_maxon, aux_maxon_position, aux_roton, aux_roton_position;

	if (20 < i && i < N - 2000)
	{
		position_m3 = dr * (double)(i - 3);
		position_m2 = dr * (double)(i - 2);
		position_m1 = dr * (double)(i - 1);
		position_0 = dr * (double)(i - 0);
		position_p1 = dr * (double)(i + 1);
		position_p2 = dr * (double)(i + 2);
		position_p3 = dr * (double)(i + 3);

		// aux it is the FIRST DERIVATIVE of the Bijl-Feynman spectrum
		aux = (-1 * (pow(position_m3, 2) / (2. * S[i - 3])) + 9 * (pow(position_m2, 2) / (2. * S[i - 2])) - 45 * (pow(position_m1, 2) / (2. * S[i - 1])) + 0 * (pow(position_0, 2) / (2. * S[i + 0])) + 45 * (pow(position_p1, 2) / (2. * S[i + 1])) - 9 * (pow(position_p2, 2) / (2. * S[i + 2])) + 1 * (pow(position_p3, 2) / (2. * S[i + 3]))) / (60 * 1.0 * dr);

		// aux it is the SECOND DERIVATIVE of the Bijl-Feynman spectrum
		aux2 = (2 * (pow(position_m3, 2) / (2. * S[i - 3])) - 27 * (pow(position_m2, 2) / (2. * S[i - 2])) + 270 * (pow(position_m1, 2) / (2. * S[i - 1])) - 490 * (pow(position_0, 2) / (2. * S[i + 0])) + 270 * (pow(position_p1, 2) / (2. * S[i + 1])) - 27 * (pow(position_p2, 2) / (2. * S[i + 2])) + 2 * (pow(position_p3, 2) / (2. * S[i + 3]))) / (180 * 1.0 * dr * dr);

		if (aux < 1e-7 && aux2 < 0)
		{
			aux_maxon_position = position_0;
			aux_maxon = pow(position_0, 2) / (2.0 * S[i]);
		}
		if (aux < 1e-9 && aux2 > 0)
		{
			aux_roton_position = position_0;
			aux_roton = pow(position_0, 2) / (2.0 * S[i]);
		}
		if (aux_maxon >= maxon[0] || aux_roton <= roton[0])
		{
			maxon[0] = aux_maxon;
			maxon_position[0] = aux_maxon_position;
			roton[0] = aux_roton;
			roton_position[0] = aux_roton_position;
		}
	}
}

__global__ void g_from_S3(
	double *__restrict__ g,
	const double *__restrict__ S,
	double *sum_diff_g)
{
	const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

	double x_shfl_src, x_shfl_dest;

	double y_val = 0.0;

#pragma unroll
	for (unsigned int m = 0; m < ((N + BLOCK_SIZE - 1) / BLOCK_SIZE); ++m)
	{
		if ((m * BLOCK_SIZE + threadIdx.x) < N)
		{
			x_shfl_src = S[threadIdx.x + m * BLOCK_SIZE] - 1.0;
			x_shfl_src *= (double)(threadIdx.x + m * BLOCK_SIZE);
		}
		else
		{
			x_shfl_src = 0.0;
		}
		__syncthreads();

		//        #pragma unroll
		for (int e = 0; e < 32; ++e)
		{
			// --- Column-major ordering - faster
			x_shfl_dest = __shfl_sync(0xffffffff, x_shfl_src, e);
			// y_val += d_j0table[tid * nCols + (e + BLOCK_SIZE * m)] * x_shfl_dest;
			y_val += j0f(double(tid) * dr * double(e + BLOCK_SIZE * m) * dk) * x_shfl_dest;
			// --- Row-major ordering - slower
			// y_val += d_V_ph_k[tid * nCols + (e + BLOCK_SIZE * m)] * x_shared[e];
		}

		__syncthreads();
	}

	if (tid < N)
	{
		sum_diff_g[0] += fabs(1.0 + (y_val * dk * dk / (2.0 * M_PI * rho)) - g[tid]);
		g[tid] = g[tid] * exp(dt * (1.0 + (y_val * dk * dk / (2.0 * M_PI * rho)) - g[tid]));
	}
}

__global__ void equal_old_S_to_S(double *old_S, const double *S)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N)
	{
		old_S[i] = S[i];
	}
}

__global__ void compute_diff_S(
	const double *__restrict__ S,
	const double *__restrict__ old_S,
	double *sum_diff_S)
{
	__shared__ double thread_calcs[MAX_BLKSZ];
	__shared__ double warp_sum_arr[WARPSZ];
	int my_i = blockDim.x * blockIdx.x + threadIdx.x;
	int my_warp = threadIdx.x / warpSize;
	int my_lane = threadIdx.x % warpSize;
	double *shared_vals = thread_calcs + my_warp * warpSize;
	double blk_result = 0.0;

	shared_vals[my_lane] = 0.0;
	if (my_i == 0 || my_i == N - 1)
	{
		shared_vals[my_lane] = 0.5 * fabs(S[my_i] - old_S[my_i]);
	}
	else
	{
		shared_vals[my_lane] = fabs(S[my_i] - old_S[my_i]);
	}

	double my_result = Shared_mem_sum(shared_vals);
	if (my_lane == 0)
	{
		warp_sum_arr[my_warp] = my_result;
	}
	__syncthreads();

	if (my_warp == 0)
	{
		if (threadIdx.x >= blockDim.x / warpSize)
		{
			warp_sum_arr[threadIdx.x] = 0.0;
		}
		blk_result = Shared_mem_sum(warp_sum_arr);
	}

	if (threadIdx.x == 0)
	{
		atomicAdd_double(sum_diff_S, blk_result);
	}
} /* Dev_trap */

__global__ void compute_diff_S2(
	const double *__restrict__ S,
	const double *__restrict__ old_S,
	double *sum_diff_S)
{
	const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

	double x_shfl_src, x_shfl_dest;

	double y_val = 0.0;

#pragma unroll
	for (unsigned int m = 0; m < ((N + BLOCK_SIZE - 1) / BLOCK_SIZE); ++m)
	{
		if ((m * BLOCK_SIZE + threadIdx.x) < N)
		{
			x_shfl_src = fabs(S[tid] - old_S[tid]);
		}
		else
		{
			x_shfl_src = 0.0;
		}
		__syncthreads();

		//        #pragma unroll
		for (int e = 0; e < 32; ++e)
		{
			// --- Column-major ordering - faster
			x_shfl_dest = __shfl_sync(0xffffffff, x_shfl_src, e);
			// y_val += d_j0table[tid * nCols + (e + BLOCK_SIZE * m)] * x_shfl_dest;
			y_val += x_shfl_dest;
			// --- Row-major ordering - slower
			// y_val += d_V_ph_k[tid * nCols + (e + BLOCK_SIZE * m)] * x_shared[e];
		}

		__syncthreads();
	}

	if (tid < N)
	{
		sum_diff_S[0] += y_val;
	}
}

void compute_MaxonRoton(const double *__restrict__ S,
						double *maxon,
						double *maxon_position,
						double *roton,
						double *roton_position)
{
	for (int i = 100; i < N - 500; i++)
	{
		double aux_maxon = 0.;
		double aux_maxon_position = 0.;
		double aux_roton = 0.;
		double aux_roton_position = 0.;

		double aux, aux2;

		double position_m3, position_m2, position_m1, position_0, position_p1, position_p2, position_p3;
		position_m3 = dr * (double)(i - 3);
		position_m2 = dr * (double)(i - 2);
		position_m1 = dr * (double)(i - 1);
		position_0 = dr * (double)(i - 0);
		position_p1 = dr * (double)(i + 1);
		position_p2 = dr * (double)(i + 2);
		position_p3 = dr * (double)(i + 3);

		// aux it is the FIRST DERIVATIVE of the Bijl-Feynman spectrum
		aux = (-1 * (pow(position_m3, 2) / (S[i - 3])) + 9 * (pow(position_m2, 2) / (S[i - 2])) - 45 * (pow(position_m1, 2) / (S[i - 1])) + 0 * (pow(position_0, 2) / (S[i + 0])) + 45 * (pow(position_p1, 2) / (S[i + 1])) - 9 * (pow(position_p2, 2) / (S[i + 2])) + 1 * (pow(position_p3, 2) / (S[i + 3]))) / (60 * 1.0 * dr);

		// aux it is the SECOND DERIVATIVE of the Bijl-Feynman spectrum
		aux2 = (2 * (pow(position_m3, 2) / (S[i - 3])) - 27 * (pow(position_m2, 2) / (S[i - 2])) + 270 * (pow(position_m1, 2) / (S[i - 1])) - 490 * (pow(position_0, 2) / (S[i + 0])) + 270 * (pow(position_p1, 2) / (S[i + 1])) - 27 * (pow(position_p2, 2) / (S[i + 2])) + 2 * (pow(position_p3, 2) / (S[i + 3]))) / (180 * 1.0 * dr * dr);

		if (aux < 1e-10 && aux2 < 0.0)
		{
			aux_maxon_position = position_0;
			aux_maxon = pow(position_0, 2) / (S[i]);
		}
		if (aux < 1e-10 && aux2 > 0.0)
		{
			aux_roton_position = position_0;
			aux_roton = pow(position_0, 2) / (S[i]);
		}

		if (aux_maxon >= maxon[0])
		{
			maxon[0] = aux_maxon;
			maxon_position[0] = aux_maxon_position;
		}

		if (aux_roton <= roton[0] && aux_roton_position >= maxon_position[0])
		{
			roton[0] = aux_roton;
			roton_position[0] = aux_roton_position;
		}
	}
}

void printer_table(double *matrix, const char *name)
{
	std::ofstream myfile;
	myfile.open(name);
	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < N; ++j)
		{
			myfile << matrix[i * N + j] << " ";
		}
		std::cout << "\n";
	}
	myfile.close();
	return;
}

void load_vector(double *vetor, int n, const char *name)
{
	std::ifstream myfile;
	myfile.open(name);
	for (int i = 0; i < n; ++i)
	{
		myfile >> vetor[i];
	}
	myfile.close();
}

__global__ void calculo_gradiente_g(double *d_g, double *d_gradiente_g)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	double aux;
	if (i >= 0 && i < N - 5)
	{

		aux = (-137. * d_g[i + 0] + 300. * d_g[i + 1] - 300. * d_g[i + 2] + 200. * d_g[i + 3] - 75. * d_g[i + 4] + 12. * d_g[i + 5]) / (60.0);

		d_gradiente_g[i] = aux / dr;
	}
	if (i > N - 6 && i < N)
	{
		d_gradiente_g[i] = (d_g[i] - d_g[i - 1]) / dr;
	}
}

__global__ void calculo_gradiente_ln_g(double *d_g, double *d_gradiente_g, double *d_gradiente_ln_g)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N)
	{
		d_gradiente_ln_g[i] = d_gradiente_g[i] * d_gradiente_g[i] / d_g[i];
	}
}

__global__ void calculate_laplaciano_Nodal(double *d_S, double *kernel, double *d_N_r)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	double sum;
	if (i < N)
	{
		sum = 0.0;
		for (int j = 0; j < N; ++j)
		{
			sum += ((double)j) * ((double)j) * ((double)j) * j0f(double(i) * dr * double(j) * dk) * (d_S[j] - 1) * (d_S[j] - 1.0) / d_S[j];
		}
		d_N_r[i] = sum * dk * dk * dk * dk / (2.0 * M_PI * rho);
	}
}

__global__ void calculo_gradiente_Nodal(double *d_g, double *d_N_r, double *d_gradiente_Nodal)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	double aux;
	if (i >= 0 && i < N - 5)
	{

		aux = (-137. * d_N_r[i + 0] + 300. * d_N_r[i + 1] - 300. * d_N_r[i + 2] + 200. * d_N_r[i + 3] - 75. * d_N_r[i + 4] + 12. * d_N_r[i + 5]) / (60.0);

		d_gradiente_Nodal[i] = aux / dr;
	}
	if (i > N - 6 && i < N)
	{
		d_gradiente_Nodal[i] = (d_N_r[i] - d_N_r[i - 1]) / dr;
	}
}

__global__ void calculo_laplaciano_ln_g(double *d_g, double *d_laplaciano_ln_g)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	double aux1, aux2;
	if (i > 0 && i < N - 7)
	{

		aux1 = ((-147.0 * d_g[i] + 360.0 * d_g[i + 1] - 450.0 * d_g[i + 2] + 400.0 * d_g[i + 3] - 225.0 * d_g[i + 4] + 72.0 * d_g[i + 5] - 10.0 * d_g[i + 6]) / (60.0));

		aux2 = (812.0 * d_g[i + 0] - 3132.0 * d_g[i + 1] + 5265.0 * d_g[i + 2] - 5080.0 * d_g[i + 3] + 2970.0 * d_g[i + 4] - 972.0 * d_g[i + 5] + 137.0 * d_g[i + 6]) / (180);

		d_laplaciano_ln_g[i] = ((-aux1 * aux1 / (d_g[i] * d_g[i])) + (aux2 / d_g[i]) + (aux1 / (d_g[i] * (double)i))) / (dr * dr);
	}
	if (i > N - 8 && i < N - 2)
	{
		d_laplaciano_ln_g[i] = ((3.0 * (d_g[i + 1]) + (d_g[i - 1]) - (d_g[i + 1] * d_g[i + 1] / d_g[i]) - d_g[i] * 3.0) + ((d_g[i + 1]) - d_g[i] * 1.0)) / (d_g[i] * dr * dr);
	}
	if (i > N - 3 && i < N - 1)
	{
		d_laplaciano_ln_g[i] = ((3.0 * (d_g[i + 1]) + (d_g[i - 1]) -
								 (d_g[i + 1] * d_g[i + 1] / d_g[i]) - d_g[i] * 3.0) +
								((d_g[i + 1]) - d_g[i] * 1.0)) /
							   (d_g[i] * dr * dr);
	}
	if (i == 0)
	{
		d_laplaciano_ln_g[i] = d_laplaciano_ln_g[1];
	}
}

__global__ void calculo_laplaciano_ln_g2(double *d_g, double *d_laplaciano_ln_g)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i > 0 && i < N - 2)
	{
		d_laplaciano_ln_g[i] = ((3.0 * (d_g[i + 1]) + (d_g[i - 1]) - (d_g[i + 1] * d_g[i + 1] / d_g[i]) - d_g[i] * 3.0) + ((d_g[i + 1]) - d_g[i] * 1.0)) / (d_g[i] * dr * dr);
	}
	if (i == 0)
	{
		d_laplaciano_ln_g[i] = d_laplaciano_ln_g[1];
	}
}

__global__ void calculo_Veff_r(double *d_g, double *d_potential, double *d_laplaciano_Nodal, double *d_laplaciano_ln_g, double *d_v_effective_r)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N)
	{
		d_v_effective_r[i] = d_g[i] * U * d_potential[i] + (-0.5) * d_g[i] * d_laplaciano_ln_g[i] + (-0.5) * d_g[i] * d_laplaciano_Nodal[i];
	}
}

__global__ void calculo_Veff_r2(double *d_g, double *d_potential, double *d_gradiente_Nodal, double *d_gradiente_g, double *d_gradiente_ln_g, double *d_v_effective_r)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N)
	{
		d_v_effective_r[i] = d_g[i] * U * d_potential[i] + (0.5) * (d_gradiente_g[i] * d_gradiente_g[i] / d_g[i]) + (-0.5) * d_gradiente_g[i] * d_gradiente_Nodal[i];
	}
}

__global__ void calculate_Veff_k(double *d_special_j0table, double *d_v_effective_r, double *v_effective_k)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	double aux;
	if (row < N)
	{
		aux = 0.0;
		for (int i = 0; i < N; i++)
		{
			aux += (double)(i)*j0f(double(row) * dr * double(i)) * (d_v_effective_r[i]);
		}
		v_effective_k[row] = aux * dr * dr * 2.0 * M_PI * rho;
	}
}

__global__ void compute_Nodal(double *d_special_j0table, double *d_S, double *d_N_r)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	double sum;
	if (i < N)
	{
		sum = 0.0;
		for (int j = 0; j < N; ++j)
		{
			sum += ((double)j) * j0f(double(i) * dr * double(j) * dk) * (d_S[j] - 1) * (d_S[j] - 1.0) / d_S[j];
		}
		d_N_r[i] = sum * dk * dk / (2.0 * M_PI * rho);
	}
}

__global__ void calculate_OBDM(double *d_special_j0table, double *d_N_r, double *d_n0, double *d_OBDM)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N)
	{
		d_OBDM[i] = d_n0[0] * exp(d_N_r[i]) / rho;
	}
}

__global__ void calculate_n0(double *d_g, double *d_N_r, double *d_n0)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	double sum = 0.0;
	if (i < N)
	{
		sum += ((double)i) * (d_g[i] + 1.0 + d_N_r[i]) - 0.5 * ((double)i) * (d_g[i] + 1.0) * d_N_r[i];
		// sum -= 0.5 * ((double)i) * (d_g[i] + 1) * d_N_r[i];
	}
	d_n0[0] = exp(sum * 2.0 * dr * dr);
}

__global__ void calculate_Veff_k2(
	const double *__restrict__ d_j0table,
	const double *__restrict__ v_effective_r,
	double *__restrict__ v_effective_k,
	const int nRows,
	const int nCols)
{
	const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

	double x_shfl_src, x_shfl_dest;

	double y_val = 0.0;

#pragma unroll
	for (unsigned int m = 0; m < ((nCols + BLOCK_SIZE - 1) / BLOCK_SIZE); ++m)
	{
		if ((m * BLOCK_SIZE + threadIdx.x) < nCols)
		{
			x_shfl_src = v_effective_r[threadIdx.x + m * BLOCK_SIZE];
			x_shfl_src *= (double)(threadIdx.x + m * BLOCK_SIZE);
		}
		else
		{
			x_shfl_src = 0.0;
		}
		__syncthreads();

		//        #pragma unroll
		for (int e = 0; e < 32; ++e)
		{
			// --- Column-major ordering - faster
			x_shfl_dest = __shfl_sync(0xffffffff, x_shfl_src, e);
			y_val += j0f(double(tid) * dr * double(e + BLOCK_SIZE * m) * dk) * x_shfl_dest;
			// --- Row-major ordering - slower
			// y_val += d_V_ph_k[tid * nCols + (e + BLOCK_SIZE * m)] * x_shared[e];
		}

		__syncthreads();
	}

	if (tid < nRows)
	{
		v_effective_k[tid] = y_val * dr * dr * 2.0 * M_PI  * rho;
	}
}

__global__ void calculate_laplaciano_Nodal2(
	const double *__restrict__ S,
	const double *__restrict__ d_j0table,
	double *__restrict__ d_laplaciano_N_r,
	const int nRows,
	const int nCols)
{
	const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

	double x_shfl_src, x_shfl_dest;

	double y_val = 0.0;

#pragma unroll
	for (unsigned int m = 0; m < ((nCols + BLOCK_SIZE - 1) / BLOCK_SIZE); ++m)
	{
		if ((m * BLOCK_SIZE + threadIdx.x) < nCols)
		{
			x_shfl_src = pow(S[threadIdx.x + m * BLOCK_SIZE] - 1.0, 2) / S[threadIdx.x + m * BLOCK_SIZE];
			x_shfl_src *= pow((double)(threadIdx.x + m * BLOCK_SIZE), 3);
		}
		else
		{
			x_shfl_src = 0.0;
		}
		__syncthreads();

		//        #pragma unroll
		for (int e = 0; e < 32; ++e)
		{
			// --- Column-major ordering - faster
			x_shfl_dest = __shfl_sync(0xffffffff, x_shfl_src, e);
			y_val += j0f(double(tid) * dr * double(e + BLOCK_SIZE * m) * dk) * x_shfl_dest;
			// --- Row-major ordering - slower
			// y_val += d_V_ph_k[tid * nCols + (e + BLOCK_SIZE * m)] * x_shared[e];
		}

		__syncthreads();
	}

	if (tid < nRows)
	{
		d_laplaciano_N_r[tid] = y_val * dk * dk * dk * dk  / (2.0 * M_PI * rho);
	}
}

__global__ void compute_N_r(
	const double *__restrict__ S,
	double *__restrict__ N_r)
{
	const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

	double x_shfl_src, x_shfl_dest;

	double y_val = 0.0;

#pragma unroll
	for (unsigned int m = 0; m < ((N + BLOCK_SIZE - 1) / BLOCK_SIZE); ++m)
	{
		if ((m * BLOCK_SIZE + threadIdx.x) < N)
		{
			x_shfl_src = pow(S[threadIdx.x + m * BLOCK_SIZE] - 1.0, 2) / (S[threadIdx.x + m * BLOCK_SIZE]);
			x_shfl_src *= (double)(threadIdx.x + m * BLOCK_SIZE);
		}
		else
		{
			x_shfl_src = 0.0;
		}
		__syncthreads();

		//        #pragma unroll
		for (int e = 0; e < 32; ++e)
		{
			// --- Column-major ordering - faster
			x_shfl_dest = __shfl_sync(0xffffffff, x_shfl_src, e);
			y_val += j0f(double(tid) * dr * double(e + BLOCK_SIZE * m) * dk) * x_shfl_dest;
			// --- Row-major ordering - slower
			// y_val += d_V_ph_k[tid * nCols + (e + BLOCK_SIZE * m)] * x_shared[e];
		}
		__syncthreads();
	}

	if (tid < N)
	{
		N_r[tid] = y_val * dk * dk / (2.0 * M_PI * rho);
	}
}

__global__ void compute_f(
	const double *__restrict__ N_r,
	const double *__restrict__ g,
	double *f)
{
	const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < N)
	{
		f[i] = sqrt(g[i] * exp(-N_r[i]));
	}
}

__global__ void compute_N_wd_k(
	double *__restrict__ N_wd_k,
	const double *__restrict__ S_wd,
	const double *__restrict__ S)
{
	const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < N)
	{
		N_wd_k[i] = (S_wd[i] - 1.0) * (1.0 - (1.0 / S[i]));
	}
}

__global__ void compute_N_wd_r(
	double *__restrict__ N_wd_r,
	const double *__restrict__ N_wd_k)
{
	const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

	double x_shfl_src, x_shfl_dest;

	double y_val = 0.0;

#pragma unroll
	for (unsigned int m = 0; m < ((N + BLOCK_SIZE - 1) / BLOCK_SIZE); ++m)
	{
		if ((m * BLOCK_SIZE + threadIdx.x) < N)
		{
			x_shfl_src = N_wd_k[threadIdx.x + m * BLOCK_SIZE];
			x_shfl_src *= (double)(threadIdx.x + m * BLOCK_SIZE);
		}
		else
		{
			x_shfl_src = 0.0;
		}
		__syncthreads();

		//        #pragma unroll
		for (int e = 0; e < 32; ++e)
		{
			// --- Column-major ordering - faster
			x_shfl_dest = __shfl_sync(0xffffffff, x_shfl_src, e);
			y_val += j0f(double(tid) * dr * double(e + BLOCK_SIZE * m) * dk) * x_shfl_dest;
			// --- Row-major ordering - slower
			// y_val += d_V_ph_k[tid * nCols + (e + BLOCK_SIZE * m)] * x_shared[e];
		}
		__syncthreads();
	}
	if (tid < N)
	{
		N_wd_r[tid] = y_val * dk * dk  / (2.0 * M_PI * rho);
	}
}

__global__ void compute_N_wd(
	double *__restrict__ N_wd,
	const double *__restrict__ S_wd,
	const double *__restrict__ S,
	const double *__restrict__ j0table,
	const int nRows,
	const int nCols)
{
	const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

	double x_shfl_src, x_shfl_dest;

	double y_val = 0.0;

#pragma unroll
	for (unsigned int m = 0; m < ((nCols + BLOCK_SIZE - 1) / BLOCK_SIZE); ++m)
	{
		if ((m * BLOCK_SIZE + threadIdx.x) < nCols)
		{
			x_shfl_src = 1.0 - (1.0 / S[threadIdx.x + m * BLOCK_SIZE]);
			x_shfl_src *= S_wd[threadIdx.x + m * BLOCK_SIZE] - 1.0;
			x_shfl_src *= (double)(threadIdx.x + m * BLOCK_SIZE);
		}
		else
		{
			x_shfl_src = 0.0;
		}
		__syncthreads();

		//        #pragma unroll
		for (int e = 0; e < 32; ++e)
		{
			// --- Column-major ordering - faster
			x_shfl_dest = __shfl_sync(0xffffffff, x_shfl_src, e);
			y_val += j0f(double(tid) * dr * double(e + BLOCK_SIZE * m) * dk) * x_shfl_dest;
			// --- Row-major ordering - slower
			// y_val += d_V_ph_k[tid * nCols + (e + BLOCK_SIZE * m)] * x_shared[e];
		}
		__syncthreads();
	}
	if (tid < nRows)
	{
		N_wd[tid] = y_val * dk * dk  / (2.0 * M_PI * rho);
	}
}

__global__ void compute_g_wd(
	double *__restrict__ g_wd,
	const double *__restrict__ f,
	const double *__restrict__ N_wd_r)
{
	const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < N)
	{
		g_wd[i] = f[i] * exp(N_wd_r[i]);
	}
}

__global__ void update_field(
	double *__restrict__ phi,
	const double *__restrict__ new_phi)
{
	const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < N)
	{
		phi[i] = (1.0 - (dt / 1.0)) * phi[i] + (dt / 1.0) * new_phi[i];
	}
}

__global__ void compute_OBDM(
	double *__restrict__ OBDM,
	const double *__restrict__ N_ww_r,
	const double *__restrict__ n_0p)
{
	const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < N)
	{
		OBDM[i] = n_0p[0] * exp(N_ww_r[i]);
	}
}

__global__ void Swd_from_gwd_and_update(
	const double *__restrict__ gwd,
	double *__restrict__ Swd,
	const double *__restrict__ d_j0table,
	const int nRows,
	const int nCols)
{
	const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

	double x_shfl_src, x_shfl_dest;

	double y_val = 0.0;

#pragma unroll
	for (unsigned int m = 0; m < ((nCols + BLOCK_SIZE - 1) / BLOCK_SIZE); ++m)
	{
		if ((m * BLOCK_SIZE + threadIdx.x) < nCols)
		{
			x_shfl_src = gwd[threadIdx.x + m * BLOCK_SIZE] - 1.0;
			x_shfl_src *= (double)(threadIdx.x + m * BLOCK_SIZE);
		}
		else
		{
			x_shfl_src = 0.0;
		}
		__syncthreads();

		//        #pragma unroll
		for (int e = 0; e < 32; ++e)
		{
			// --- Column-major ordering - faster
			x_shfl_dest = __shfl_sync(0xffffffff, x_shfl_src, e);
			y_val += j0f(double(tid) * dr * double(e + BLOCK_SIZE * m) * dk) * x_shfl_dest;
			// --- Row-major ordering - slower
			// y_val += d_V_ph_k[tid * nCols + (e + BLOCK_SIZE * m)] * x_shared[e];
		}
		__syncthreads();
	}

	if (tid < nRows)
	{
		Swd[tid] = (1.0 - dt / 10) * Swd[tid] + (dt / 10) * (1.0 + y_val * dr * dr * 2.0 * M_PI  * rho);
	}
}

__global__ void Dev_trap(
	const double *g /* in       */,
	const double *Ng /* in       */,
	double *trap_p /* in/out   */)
{
	__shared__ double thread_calcs[MAX_BLKSZ];
	__shared__ double warp_sum_arr[WARPSZ];
	int my_i = blockDim.x * blockIdx.x + threadIdx.x;
	int my_warp = threadIdx.x / warpSize;
	int my_lane = threadIdx.x % warpSize;
	double *shared_vals = thread_calcs + my_warp * warpSize;
	double blk_result = 0.0;

	shared_vals[my_lane] = 0.0;
	if (my_i == 0 || my_i == N - 1)
	{
		shared_vals[my_lane] = 0.5 * ((g[my_i] - 1.0 - Ng[my_i]) - 0.5 * (g[my_i] - 1.0) * Ng[my_i]) * (double)(my_i);
	}
	else
	{
		shared_vals[my_lane] = ((g[my_i] - 1.0 - Ng[my_i]) - 0.5 * (g[my_i] - 1.0) * Ng[my_i]) * (double)(my_i);
	}

	double my_result = Shared_mem_sum(shared_vals);
	if (my_lane == 0)
	{
		warp_sum_arr[my_warp] = my_result;
	}
	__syncthreads();

	if (my_warp == 0)
	{
		if (threadIdx.x >= blockDim.x / warpSize)
		{
			warp_sum_arr[threadIdx.x] = 0.0;
		}
		blk_result = Shared_mem_sum(warp_sum_arr);
	}

	if (threadIdx.x == 0)
	{
		atomicAdd_double(trap_p, blk_result * dr * dr * (2.0 * M_PI  * rho));
	}
} /* Dev_trap */

__global__ void compute_N_ww_r(
	const double *__restrict__ N_wd_k,
	const double *__restrict__ S_wd,
	double *__restrict__ N_ww_r)
{
	const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

	double x_shfl_src, x_shfl_dest;

	double y_val = 0.0;

#pragma unroll
	for (unsigned int m = 0; m < ((N + BLOCK_SIZE - 1) / BLOCK_SIZE); ++m)
	{
		if ((m * BLOCK_SIZE + threadIdx.x) < N)
		{
			x_shfl_src = S_wd[threadIdx.x + m * BLOCK_SIZE] - 1.0 - N_wd_k[threadIdx.x + m * BLOCK_SIZE];
			x_shfl_src *= S_wd[threadIdx.x + m * BLOCK_SIZE] - 1.0;
			x_shfl_src *= (double)(threadIdx.x + m * BLOCK_SIZE);
		}
		else
		{
			x_shfl_src = 0.0;
		}
		__syncthreads();

		//        #pragma unroll
		for (int e = 0; e < 32; ++e)
		{
			// --- Column-major ordering - faster
			x_shfl_dest = __shfl_sync(0xffffffff, x_shfl_src, e);
			y_val += j0f(double(tid) * dr * double(e + BLOCK_SIZE * m) * dk) * x_shfl_dest;
			// --- Row-major ordering - slower
			// y_val += d_V_ph_k[tid * nCols + (e + BLOCK_SIZE * m)] * x_shared[e];
		}
		__syncthreads();
	}

	if (tid < N)
	{
		N_ww_r[tid] = y_val * dk * dk / (2.0 * M_PI * rho);
	}
}

__global__ void compute_N_wd_r_and_update_(
	const double *__restrict__ S,
	const double *__restrict__ S_wd,
	double *__restrict__ N_wd_r,
	const double *__restrict__ j0table,
	const int nRows,
	const int nCols)
{
	const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

	double x_shfl_src, x_shfl_dest;

	double y_val = 0.0;

	#pragma unroll
	for (unsigned int m = 0; m < ((nCols + BLOCK_SIZE - 1) / BLOCK_SIZE); ++m)
	{
		if ((m * BLOCK_SIZE + threadIdx.x) < nCols)
		{
			x_shfl_src = 1.0 - (1.0 / S[threadIdx.x + m * BLOCK_SIZE]);
			x_shfl_src *= S_wd[threadIdx.x + m * BLOCK_SIZE] - 1.0;
			x_shfl_src *= (double)(threadIdx.x + m * BLOCK_SIZE);
		}
		else
		{
			x_shfl_src = 0.0;
		}
		__syncthreads();

		#pragma unroll
		for (int e = 0; e < 32; ++e)
		{
			// --- Column-major ordering - faster
			x_shfl_dest = __shfl_sync(0xffffffff, x_shfl_src, e);
			y_val += j0f(double(tid) * dr * double(e + BLOCK_SIZE * m) * dk) * x_shfl_dest;
			// --- Row-major ordering - slower
			// y_val += d_V_ph_k[tid * nCols + (e + BLOCK_SIZE * m)] * x_shared[e];
		}
		__syncthreads();
	}

	if (tid < nRows)
	{
		N_wd_r[tid] = (1.0 - dt / 50) * N_wd_r[tid] + (dt / 50) * (1.0 + y_val * dr * dr * 2.0 * M_PI  * rho);
	}
}

__global__ void general_update(
	double *__restrict__ N_wd,
	double *__restrict__ g_wd,
	double *__restrict__ S_wd,
	const double *__restrict__ new_N_wd,
	const double *__restrict__ new_g_wd,
	const double *__restrict__ new_S_wd)
{
	const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < N)
	{
		N_wd[i] = (1.0 - dt / 10) * N_wd[i] + (dt / 10) * new_N_wd[i];
		S_wd[i] = (1.0 - dt / 10) * S_wd[i] + (dt / 10) * new_S_wd[i];
		g_wd[i] = (1.0 - dt / 10) * g_wd[i] + (dt / 10) * new_g_wd[i];
	}
}

__global__ void compute_new_N_wd_k(
	double *__restrict__ new_N_wd,
	const double *__restrict__ S_wd,
	const double *__restrict__ S)
{
	const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < N)
	{
		new_N_wd[i] = (S_wd[i] - 1.0) * (1.0 - (1.0 / S[i]));
	}
}

__global__ void N_wdr_from_N_wd(
	double *__restrict__ N_wd_r,
	const double *__restrict__ N_wd,
	const double *__restrict__ d_j0table,
	const int nRows,
	const int nCols)
{
	const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

	double x_shfl_src, x_shfl_dest;

	double y_val = 0.0;

#pragma unroll
	for (unsigned int m = 0; m < ((nCols + BLOCK_SIZE - 1) / BLOCK_SIZE); ++m)
	{
		if ((m * BLOCK_SIZE + threadIdx.x) < nCols)
		{
			x_shfl_src = N_wd[threadIdx.x + m * BLOCK_SIZE];
			x_shfl_src *= (double)(threadIdx.x + m * BLOCK_SIZE);
		}
		else
		{
			x_shfl_src = 0.0;
		}
		__syncthreads();

		//        #pragma unroll
		for (int e = 0; e < 32; ++e)
		{
			// --- Column-major ordering - faster
			x_shfl_dest = __shfl_sync(0xffffffff, x_shfl_src, e);
			// y_val += d_j0table[tid * nCols + (e + BLOCK_SIZE * m)] * x_shfl_dest;
			y_val += j0f(double(tid) * dr * double(e + BLOCK_SIZE * m) * dk) * x_shfl_dest;
			// --- Row-major ordering - slower
			// y_val += d_V_ph_k[tid * nCols + (e + BLOCK_SIZE * m)] * x_shared[e];
		}

		__syncthreads();
	}

	if (tid < nRows)
	{
		N_wd_r[tid] = y_val * dk * dk  / (2.0 * M_PI * rho);
	}
}

__global__ void compute_nq(
	double *__restrict__ nq,
	const double *__restrict__ OBDM,
	const double *__restrict__ n_0p)
{
	const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

	double x_shfl_src, x_shfl_dest;

	double y_val = 0.0;

#pragma unroll
	for (unsigned int m = 0; m < ((N + BLOCK_SIZE - 1) / BLOCK_SIZE); ++m)
	{
		if ((m * BLOCK_SIZE + threadIdx.x) < N)
		{
			x_shfl_src = OBDM[threadIdx.x + m * BLOCK_SIZE] - n_0p[0];
			x_shfl_src *= (double)(threadIdx.x + m * BLOCK_SIZE);
		}
		else
		{
			x_shfl_src = 0.0;
		}
		__syncthreads();

		//        #pragma unroll
		for (int e = 0; e < 32; ++e)
		{
			// --- Column-major ordering - faster
			x_shfl_dest = __shfl_sync(0xffffffff, x_shfl_src, e);
			y_val += j0f(double(tid) * dr * double(e + BLOCK_SIZE * m) * dk) * x_shfl_dest;
			// --- Row-major ordering - slower
			// y_val += d_V_ph_k[tid * nCols + (e + BLOCK_SIZE * m)] * x_shared[e];
		}
		__syncthreads();
	}

	if (tid < N)
	{
		nq[tid] = y_val * dr * dr * 2.0 * M_PI  * rho;
	}
}

/* Host code */
void compute_Trap_wrapper(
	const double *g_wd /* in   */,
	const double *N_wd /* in   */,
	double *d_trap_p /* out  */)
{
	Dev_trap<<<Blocks_N, ThreadsPerBlock_N>>>(g_wd, N_wd, d_trap_p);
	cudaDeviceSynchronize();
} /* Trap_wrapper */


