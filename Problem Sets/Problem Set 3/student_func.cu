/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"

__global__
void get_minmax(const float* d_logLuminance, float *d_out, int size, bool is_max){
   int indx = blockIdx.x * blockDim.x + threadIdx.x;
   int tindx = threadIdx.x;

   extern __shared__ float sh_mem[];

   if(indx > size)
      return;

   sh_mem[tindx] = d_logLuminance[indx];
   __syncthreads();

   for(int s = blockDim.x / 2; s > 0; s >>= 1){
      if(tindx < s){
         if(is_max)
            sh_mem[tindx] = fmax(sh_mem[tindx], sh_mem[tindx + s]);
         else
            sh_mem[tindx] = fmin(sh_mem[tindx], sh_mem[tindx + s]);
      }
      __syncthreads();
   }

   if(tindx == 0)
      d_out[blockIdx.x] = sh_mem[tindx];
}

float maxmin(const float *d_logLuminance, int numRows, int numCols, bool is_max){
   int THREADS_PER_BLOCK = numRows;
   int size = numCols * numRows;
   dim3 blockSize(THREADS_PER_BLOCK);
   dim3 gridSize((THREADS_PER_BLOCK + size - 1) / THREADS_PER_BLOCK);

   float *d_out, *d_int;
   cudaMalloc(&d_out, sizeof(float) * numRows * numCols);
   cudaMalloc(&d_int, sizeof(float) * numRows * numCols);

   get_minmax<<<gridSize, blockSize, THREADS_PER_BLOCK * sizeof(float)>>>(d_logLuminance, d_int, size, is_max);
   get_minmax<<<1, blockSize, THREADS_PER_BLOCK * sizeof(float)>>>(d_int, d_out, size, is_max);

   cudaDeviceSynchronize();

   float ans;
   cudaMemcpy(&ans, d_out, sizeof(float), cudaMemcpyDeviceToHost);

   cudaFree(d_out);
   cudaFree(d_int);

   return ans;
}

__global__
void hist(const float* logLuminance, int *d_hist, const size_t numBins, const int size, const int min_logLum, const int lumRange){
   
   int indx = blockIdx.x * blockDim.x + threadIdx.x;

   if(indx > size)
      return;

   int bin = ( (logLuminance[indx] - min_logLum) * numBins )/ lumRange;
   atomicAdd(&d_hist[bin], 1);
   //printf("%d ", bin);
}

//--------HILLIS-STEELE SCAN----------
//Optimal step efficiency (histogram is a relatively small vector)
//Works on maximum 1024 (Pascal) elems vector.
__global__ void scan_hillis_steele(unsigned int* d_out,const int* d_in, int size) {
	extern __shared__ unsigned int temp[];
	int tid = threadIdx.x;
	int pout = 0,pin=1;
	temp[tid] = tid>0? d_in[tid-1]:0; //exclusive scan
	__syncthreads();

	//double buffered
	for (int off = 1; off < size; off <<= 1) {
		pout = 1 - pout;
		pin = 1 - pout;
		if (tid >= off) temp[size*pout + tid] = temp[size*pin + tid]+temp[size*pin + tid - off];
		else temp[size*pout + tid] = temp[size*pin + tid];
		__syncthreads();
	}
	d_out[tid] = temp[pout*size + tid];
}

void histogram_prefixsum(const float* d_logLuminance, unsigned int* d_cdf, const size_t numBins, const int min_logLum, 
                  const int lumRange, const size_t numRows, const size_t numCols){

   int THREADS_PER_BLOCK = numRows;
   int size = numCols * numRows;
   dim3 blockSize(THREADS_PER_BLOCK);
   dim3 gridSize((THREADS_PER_BLOCK + size - 1) / THREADS_PER_BLOCK);

   int *d_hist;
   cudaMalloc(&d_hist, sizeof(int) * numBins);

   hist<<<gridSize, blockSize>>>(d_logLuminance, d_hist, numBins, size, min_logLum, lumRange);
   cudaDeviceSynchronize();
   scan_hillis_steele<<<1, numBins, 2*numBins*sizeof(unsigned int)>>>(d_cdf, d_hist, numBins);
   cudaDeviceSynchronize();

   int h_hist[numBins];
   cudaMemcpy(h_hist, d_cdf, sizeof(int) * numBins, cudaMemcpyDeviceToHost);
   
   for(int i = 0; i < numBins; i++){
      if(h_hist[i] != 0)
         printf("Bin %d : %d\n", i, h_hist[i]);
   }
   
   cudaFree(d_hist);
}



void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */
      printf("%d\n", numCols * numRows);
      max_logLum = maxmin(d_logLuminance, numRows, numCols, true);
      min_logLum = maxmin(d_logLuminance, numRows, numCols, false);
      
      printf("GPU max : %f\n", max_logLum);
      printf("GPU min : %f\n", min_logLum);

      float lumRange = max_logLum - min_logLum;
      
      histogram_prefixsum(d_logLuminance, d_cdf, numBins, min_logLum, lumRange, numRows, numCols);
}
