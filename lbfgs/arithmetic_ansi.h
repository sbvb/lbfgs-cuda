/*
 *      ANSI C implementation of vector operations.
 *
 * Copyright (c) 2007-2010 Naoaki Okazaki
 * All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

/* $Id$ */

#include <stdlib.h>
#include <memory.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <time.h>

#if     LBFGS_FLOAT == 32 && LBFGS_IEEE_FLOAT
#define fsigndiff(x, y) (((*(uint32_t*)(x)) ^ (*(uint32_t*)(y))) & 0x80000000U)
#else
#define fsigndiff(x, y) (*(x) * (*(y) / fabs(*(y))) < 0.)
#endif/*LBFGS_IEEE_FLOAT*/

extern cublasHandle_t handle;

extern lbfgsfloatval_t *d_x;
extern lbfgsfloatval_t *d_y;
extern lbfgsfloatval_t *d_result;
extern double alpha;

extern int cvecset;
extern int cveccpy; 
extern int cvecncpy;
extern int cvecadd;
extern int cvecdiff;
extern int cvecscale; 
extern int cvecmul;
extern int cvecdot;
extern int cmy_vecdot;
extern int cmy_vec_copy_Add;

extern int timevecset ;
extern int timeveccpy ;
extern int timevecncpy ;
extern int timevecadd ;
extern int timevecdiff ;
extern int timevecscale ;
extern int timevecmul ;
extern int timevecdot ;
extern int timemy_vecdot ;
extern int timemy_vec_copy_Add ;



inline static void* vecalloc(size_t size)
{
    void *memblock = malloc(size);
    if (memblock) {
        memset(memblock, 0, size);
    }
    return memblock;
}

inline static void vecfree(void *memblock)
{
    free(memblock);
}

inline static void vecset(lbfgsfloatval_t *x, const lbfgsfloatval_t c, const int n)
{  
	//cvecset++;
	cublasSetVector(n, sizeof(lbfgsfloatval_t), &c, 1, d_x, 1); 
	/*
		int i;   

		for (i = 0;i < n;++i) {
			x[i] = c;
		}
	}*/
}

inline static void veccpy(lbfgsfloatval_t *y, const lbfgsfloatval_t *x, const int n)
{       
    //cveccpy++;	
	//clock_t time = clock();
	
	cublasSetVector(n, sizeof(lbfgsfloatval_t), x, 1, d_x, 1);

	cublasDcopy(handle, n, d_x, 1, d_y, 1);

	cublasGetVector(n, sizeof(y[0]), d_y, 1, y, 1);    
	
	//time = clock() - time;	
    //timeveccpy += (double)time / CLOCKS_PER_SEC * 1000;
	
	
	/*
	   int i;

		for (i = 0;i < n;++i) {
			y[i] = x[i];
		}
	}
	*/
}

inline static void vecncpy(lbfgsfloatval_t *y, const lbfgsfloatval_t *x, const int n)
{   
	//cvecncpy++;	
	//clock_t time = clock();
	
	cudaMemset(d_y, 0, n*sizeof(lbfgsfloatval_t)); 

	cublasSetVector(n, sizeof(lbfgsfloatval_t), x, 1, d_x, 1);    

	cublasDaxpy(handle, n, &alpha, d_x, 1, d_y, 1);

	cublasGetVector(n, sizeof(y[0]), d_y, 1, y, 1);
	
	//time = clock() - time;	
    //timevecncpy += (double)time / CLOCKS_PER_SEC * 1000;
	/*
	 
		int i;

		for (i = 0;i < n;++i) {
			y[i] = -x[i];
		}
	}*/

}

inline static void vecadd(lbfgsfloatval_t *y, const lbfgsfloatval_t *x, const lbfgsfloatval_t c, const int n)
{   
	//cvecadd++;	
	//clock_t time = clock();	
	
	/*
		cublasSetVector(n, sizeof(lbfgsfloatval_t), x, 1, d_x, 1);
		cublasSetVector(n, sizeof(lbfgsfloatval_t), y, 1, d_y, 1);

		cublasDaxpy(handle, n, &c, d_x, 1, d_y, 1);

		cublasGetVector(n, sizeof(y[0]), d_y, 1, y, 1);    
	*/
		int i;

		for (i = 0;i < n;++i) {
			y[i] += c * x[i];
		}
	
	//time = clock() - time;	
    //timevecadd += (double)time / CLOCKS_PER_SEC * 1000;
	
}

inline static void vecdiff(lbfgsfloatval_t *z, const lbfgsfloatval_t *x, const lbfgsfloatval_t *y, const int n)
{   
	//cvecdiff++;	
	//clock_t time = clock();
	/*
		cublasSetVector(n, sizeof(lbfgsfloatval_t), x, 1, d_x, 1);
		cublasSetVector(n, sizeof(lbfgsfloatval_t), y, 1, d_y, 1);

		cublasDcopy(handle, n, d_x, 1, d_result, 1);	
		cublasDaxpy(handle, n, &alpha, d_y, 1, d_result, 1);	

		cublasGetVector(n, sizeof(y[0]), d_result, 1, z, 1);    
	*/		
		int i;

		for (i = 0;i < n;++i) {
			z[i] = x[i] - y[i];
		}
		
	//time = clock() - time;	
    //timevecdiff += (double)time / CLOCKS_PER_SEC * 1000;
		
}

inline static void vecscale(lbfgsfloatval_t *y, const lbfgsfloatval_t c, const int n)
{   
	//cvecscale++;	
	//clock_t time = clock();	
	
	cublasSetVector(n, sizeof(lbfgsfloatval_t), y, 1, d_y, 1);
	
	cublasDscal(handle, n, &c, d_y, 1);
	
	cublasGetVector(n, sizeof(y[0]), d_y, 1, y, 1);
	/*
		int i;

		for (i = 0;i < n;++i) {
			y[i] *= c;
		}
	*/
	
	//time = clock() - time;	
    //timevecscale += (double)time / CLOCKS_PER_SEC * 1000;
	
}

inline static void vecmul(lbfgsfloatval_t *y, const lbfgsfloatval_t *x, const int n)
{    
	//cvecmul++;	
	//clock_t time = clock();	
	
	cublasSetVector(n, sizeof(lbfgsfloatval_t), x, 1, d_x, 1);
	cublasSetVector(n, sizeof(lbfgsfloatval_t), y, 1, d_y, 1);
	
	cublasDdot(handle, n, d_x, 1, d_y, 1, y);    
	/*
		int i;

		for (i = 0;i < n;++i) {
			y[i] *= x[i];
		}
	*/
	//time = clock() - time;	
    //timevecmul += (double)time / CLOCKS_PER_SEC * 1000;
}

inline static void vecdot(lbfgsfloatval_t* s, const lbfgsfloatval_t *x, const lbfgsfloatval_t *y, const int n)
{       
	//cvecdot++;	
	//clock_t time = clock();	
	
	cublasSetVector(n, sizeof(lbfgsfloatval_t), x, 1, d_x, 1);
	cublasSetVector(n, sizeof(lbfgsfloatval_t), y, 1, d_y, 1);
	
	cublasDdot(handle, n, d_x, 1, d_y, 1, s);    
	/*

		int i;
		*s = 0.;
		for (i = 0;i < n;++i) {
			*s += x[i] * y[i];
		}
	*/
	//time = clock() - time;	
    //timevecdot += (double)time / CLOCKS_PER_SEC * 1000;
}

/**
@param s1 - output - vector - s1[i] = x[i] * y[i]
@param s2 - output - vector - s2[i] = y[i] * y[i] 
@param x - input - vector
@param y - input - vector
@param n - input - dimension
*/
inline static void my_vecdot(lbfgsfloatval_t* s1, lbfgsfloatval_t* s2, const lbfgsfloatval_t *x, const lbfgsfloatval_t *y, const int n)
{       
	//cmy_vecdot++;
	//clock_t time = clock();
	
	cublasSetVector(n, sizeof(lbfgsfloatval_t), x, 1, d_x, 1);
	cublasSetVector(n, sizeof(lbfgsfloatval_t), y, 1, d_y, 1);
	
	cublasDdot(handle, n, d_x, 1, d_y, 1, s1);   	
	cublasDdot(handle, n, d_y, 1, d_y, 1, s2);    
		
/*
		int i;
		*s1 = 0.;
		for (i = 0;i < n;++i) {
			*s1 += x[i] * y[i];
		}
	}*/
	
	//time = clock() - time;	
    //timemy_vecdot += (double)time / CLOCKS_PER_SEC * 1000;
	
}

/**
@param x - output - x = xp + s * stp 
@param xp - input - vector
@param s - input - vector
@param stp - input - real parameter
@param n - input - dimension
*/
inline static void my_vec_copy_Add(lbfgsfloatval_t *x, const lbfgsfloatval_t *xp, lbfgsfloatval_t *s, lbfgsfloatval_t stp, const int n)
{
	//cmy_vec_copy_Add++;	
	//clock_t time = clock();	
	
	cublasSetVector(n, sizeof(lbfgsfloatval_t), xp, 1, d_result, 1); // d_result = xp (copy xp from host to device)	
	cublasSetVector(n, sizeof(lbfgsfloatval_t), s, 1, d_y, 1); // d_y = s (copy s from host to device)

	cublasDaxpy(handle, n, &stp, d_y, 1, d_result, 1); // d_result = d_y * stp (inside device)

	cublasGetVector(n, sizeof(lbfgsfloatval_t), d_result, 1, x, 1); // copy from device to host x = d_result
	/*
		int i;

		for (i = 0;i < n;++i) {
			y[i] += c * x[i];
		}    
	}*/	
	//time = clock() - time;	
    //timemy_vec_copy_Add += (double)time / CLOCKS_PER_SEC * 1000;
	
}

inline static void vec2norm(lbfgsfloatval_t* s, const lbfgsfloatval_t *x, const int n)
{
    vecdot(s, x, x, n);
    *s = (lbfgsfloatval_t)sqrt(*s);
}

inline static void vec2norminv(lbfgsfloatval_t* s, const lbfgsfloatval_t *x, const int n)
{
    vec2norm(s, x, n);
    *s = (lbfgsfloatval_t)(1.0 / *s);
}
