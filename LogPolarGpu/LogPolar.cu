#include "GpuLogPolar.h"

static float GPUmilliseconds;
static cudaEvent_t start, stop;
int rowsPerKernel = 1;
int angleLimit = 360;

#define BLOCK_SIZE_X	16
#define BLOCK_SIZE_Y	32
#ifndef FLT_MAX
#define FLT_MAX         1e30
#endif

__global__ void kernelGetLpImage(const MonoImage devBgImage, MonoImage devLpPatchImage, const LpTransformEngine<float> devEngine, Simple2dPoint<int> patchPosition) {
	// Block row and column
	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;

	int row = threadIdx.y;
	int col = threadIdx.x;

	int x = blockCol * blockDim.x + col;
	int y = blockRow * blockDim.y + row;

	if (x < devLpPatchImage.getWidth() && y < devLpPatchImage.getHeight()) {
		Simple2dPoint<float> point = devEngine.getTransformCords(x, y);
		Simple2dPoint<int> position = { 0, 0 };
		devLpPatchImage.setPixel(x, y, interpolate(devBgImage, point, position));
	}

	__syncthreads();
}

__device__ uint8_t interpolate(const MonoImage devBgImage, Simple2dPoint<float> point, Simple2dPoint<int> position) {
	int x1, x2, y1, y2;
	x1 = (int)point.x + position.x;
	x2 = x1 + 1;
	y1 = (int)point.y + position.y;
	y2 = y1 + 1;
	float dx, dy, a, b, c, d, ab, dc;
	a = (float)(devBgImage.getPixel(x1, y1));   //  A----B
	b = (float)(devBgImage.getPixel(x2, y1));   //  |    |
	c = (float)(devBgImage.getPixel(x2, y2));   //  D----C
	d = (float)(devBgImage.getPixel(x1, y2));
	dx = point.x - (float)((int)point.x);
	dy = point.y - (float)((int)point.y);
	ab = b > a ? a + dx * (b - a) : a - dx * (a - b);
	dc = c > d ? d + dx * (c - d) : d - dx * (d - c);
	return (uint8_t)(dc > ab ? (ab + dy * (dc - ab)) : (ab - dy * (ab - dc)));
}

__global__ void kernelSearch4dSpace(const MonoImage devBgImage, const MonoImage devLpPatchImage, const LpTransformEngine<float> devEngine, int startRow, Results *devSearchResults, Simple2dPoint<int> patchSize, int angLim) {

	// Block identifies patch image position within reference image
	int patchColPosition = blockIdx.x;
	int patchRowPosition = startRow + blockIdx.y;
	Simple2dPoint<int> patchPosition = { patchColPosition, patchRowPosition };

	int patchCols = devLpPatchImage.getWidth();
	int patchRows = devLpPatchImage.getHeight();

	// limit possible shift to avoid exceeding reference image
	int maxPatchColPosition = devBgImage.getWidth() - patchCols;
	int maxPatchRowPosition = devBgImage.getHeight() - patchRows;
	if (patchColPosition >= maxPatchColPosition || patchRowPosition >= maxPatchRowPosition) return;

	// Thread identifies LP patch column and scale shift in LP patch image also
	int patchCol = threadIdx.x;

	// limit thread number to number of patch image cols
	if (patchCol >= patchCols) return;

	// Each block stores LP of reference image fragment and LP of patch
	extern __shared__ uint8_t sharedArray[];
	float *corelationValues = (float*) sharedArray;									// array of [patchSize.x] float elements
	uint8_t *sharedLpBgPatch = sharedArray + patchSize.x * sizeof(float);			// array of [patchSize.x*patchSize.y] uint8_t elements
	uint8_t *sharedLpPatch = sharedLpBgPatch + (patchSize.x * patchSize.y);			// array of [patchSize.x*patchSize.y] uint8_t elements
	uint8_t *corelationBestAngle = sharedLpPatch + (patchSize.x * patchSize.y);		// array of [patchSize.x] uint8_t elements

	// Each thread within a block computes one column of LP patch
	for (int row = 0; row < patchRows; row++) {
		sharedLpBgPatch[row * patchCols + patchCol] = interpolate(devBgImage, devEngine.getTransformCords(patchCol, row), patchPosition);
		sharedLpPatch[row * patchCols + patchCol] = devLpPatchImage.getPixel(patchCol, row);
	}
	__syncthreads(); // make sure LP is complete before further computations

	// Now, each thread computes corelations for one shift in scale axis
	int commonCols, bgOffset, patchOffset;
	if (patchCol < patchCols / 2) {
		commonCols = patchCols / 2 + patchCol;
		bgOffset = 0;
		patchOffset = patchCols / 2 - patchCol;
	}
	else {
		commonCols = 3 * patchCols / 2 - patchCol;
		bgOffset = patchCol - patchCols / 2;
		patchOffset = 0;
	}

#if (CORREL_MEASURE == COV_VAR)
	float lpBgPatchMean, lpPatchMean;
	// Compute mean value for lpBgPatch and lpPatch
	lpBgPatchMean = getMeanValue(sharedLpBgPatch, patchCols, patchRows, commonCols, bgOffset);
	lpPatchMean = getMeanValue(sharedLpPatch, patchCols, patchRows, commonCols, patchOffset);
#endif

	
	float bestScore, currentScore;
	bestScore = CORREL_MEASURE ? FLT_MAX : 0.0;
	currentScore = bestScore;

	int bestAngleIndex = 0;
	int range = (angLim * patchRows) / 360;
	// Compute corelation for each angle shift
	for (int a = patchRows - range / 2, angle = a; a <= patchRows + range / 2; angle = a % patchRows, a++) {
#if (CORREL_MEASURE == COV_VAR)
		currentScore = computeCorelation(sharedLpBgPatch, sharedLpPatch, patchCols, patchRows, lpBgPatchMean, lpPatchMean, angle, commonCols, bgOffset, patchOffset);
		if (currentScore > bestScore) {
			bestAngleIndex = angle;
			bestScore = currentScore;
		}
#elif (CORREL_MEASURE == SAD)
		currentScore = computeSADScore(sharedLpBgPatch, sharedLpPatch, patchCols, patchRows, angle, commonCols, bgOffset, patchOffset);
		if (currentScore < bestScore) {
			bestAngleIndex = angle;
			bestScore = currentScore;
		}
#elif (CORREL_MEASURE == CENSUS) 
		currentScore = computeCensusScore(sharedLpBgPatch, sharedLpPatch, patchCols, patchRows, angle, commonCols, bgOffset, patchOffset);
		if (currentScore < bestScore) {
			bestAngleIndex = angle;
			bestScore = currentScore;
		}
#endif
	}

	corelationValues[patchCol] = bestScore;
	corelationBestAngle[patchCol] = (unsigned char)bestAngleIndex;
	__syncthreads();

	// first thread in each block chooses the best corelation result, and scale and angle for this result
	float blockBestScore = CORREL_MEASURE ? FLT_MAX : 0.0;
	int bestScale = 0;
	if (patchCol == 0) {
		for (int i = 0; i < patchCols; i++) {
			if (CORREL_MEASURE ? corelationValues[i] < blockBestScore : corelationValues[i] > blockBestScore) {
				bestScale = i;
				blockBestScore = corelationValues[i];
			}
		}
		(devSearchResults + patchRowPosition * maxPatchColPosition + patchColPosition)->scale = bestScale;
		(devSearchResults + patchRowPosition * maxPatchColPosition + patchColPosition)->angle = corelationBestAngle[bestScale];
		(devSearchResults + patchRowPosition * maxPatchColPosition + patchColPosition)->result = blockBestScore;// corelationValues[bestScale];
	}
	__syncthreads();
}

CUDA_HOSTDEV float getMeanValue(uint8_t *buffer, int cols, int rows, int commonCols, int offset) {
	int sum = 0;
	for (int row = 0; row < rows; row++) {
		for (int col = 0; col < commonCols; col++) {
			sum += *(buffer + row * cols + col + offset);
		}
	}
	return (float)sum / (float)(commonCols * rows);
}

CUDA_HOSTDEV float computeCorelation(	const uint8_t *lpBgPatchBuf, const uint8_t *lpPatchBuf, 
										int cols, int rows, float lpBgPatchMean, float lpPatchMean, 
										int angleShift, int commonCols, int bgOffset, int patchOffset) {
	float tmp1, tmp2, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
	for (int row = 0; row < rows; row++) {
		for (int col = 0; col < commonCols; col++) {
			tmp1 = (float)(*(lpBgPatchBuf + row * cols + col + bgOffset)) - lpBgPatchMean;
			tmp2 = (float)(*(lpPatchBuf + (((row + angleShift) < rows) ? (row + angleShift) : (row + angleShift - rows)) * cols + col + patchOffset)) - lpPatchMean;
			sum1 += tmp1 * tmp2;
			sum2 += tmp1 * tmp1;
			sum3 += tmp2 * tmp2;
		}
	}
	float score = (sum2 * sum3 == 0.0) ? 0.0 : sum1 / sqrt(sum2 * sum3);
	return score;
}

CUDA_HOSTDEV float computeSADScore( const uint8_t *lpBgPatchBuf, const uint8_t *lpPatchBuf, 
									int cols, int rows, int angleShift, int commonCols, int bgOffset, int patchOffset) {
	float tmp1, tmp2, sum = 0.0;
	for (int row = 0; row < rows; row++) {
		for (int col = 0; col < commonCols; col++) {
			tmp1 = (float)(*(lpBgPatchBuf + row * cols + col + bgOffset));
			tmp2 = (float)(*(lpPatchBuf + (((row + angleShift) < rows) ? (row + angleShift) : (row + angleShift - rows)) * cols + col + patchOffset));
			sum += fabs(tmp1 - tmp2);
		}
	}
	return sum / (float)(commonCols);
}

__device__ float computeCensusScore( const uint8_t *lpBgPatchBuf, const uint8_t *lpPatchBuf,
						  int cols, int rows, int angleShift, int commonCols, int bgOffset, int patchOffset) {
	long hamming_sum = 0;
	uint8_t xor_val, tmp1, tmp2;

	for (int row = 0; row < rows; row++) {
		for (int col = 0; col < commonCols; col++) {
			tmp1 = *(lpBgPatchBuf + row * cols + col + bgOffset);
			tmp2 = *(lpPatchBuf + (((row + angleShift) < rows) ? (row + angleShift) : (row + angleShift - rows)) * cols + col + patchOffset);
			xor_val = tmp1 ^ tmp2;
			if (xor_val != 0){
				hamming_sum += __popc(xor_val);
			}
		}
	}

	return (float)(hamming_sum) / (float)(commonCols);
}

void gpuGetLpImage(const MonoImage & bgImage, const MonoImage & patchImage, const LpTransformEngine<float> & engine, Results *sliceResult, Simple2dPoint<int> *slicePos) {
	
	cudaError_t cudaStatus = cudaSuccess;
	int bgWidth = bgImage.getWidth();
	int bgHeight = bgImage.getHeight();
	int patchWidth = patchImage.getWidth();
	int patchHeight = patchImage.getHeight();
	size_t bgSize = bgWidth * bgHeight * sizeof(uint8_t);
	size_t patchSize = patchWidth * patchHeight * sizeof(uint8_t);
	size_t engineSize = patchWidth * patchHeight * sizeof(Simple2dPoint<float>);

	Results *results;
	results = (Results*)malloc((bgWidth - patchWidth) * (bgHeight - patchHeight) * sizeof(Results));
	MonoImage devLpPatchImage(patchWidth, patchHeight);
	cudaMalloc(&devLpPatchImage.imageData, patchSize);
	MonoImage lpPatchImage(patchWidth, patchHeight);
	lpPatchImage.imageData = (uint8_t*)malloc(patchSize);

	Results *devSearchResults;
	cudaMalloc(&devSearchResults, (bgWidth - patchWidth) * (bgHeight - patchHeight) * sizeof(Results));

	MonoImage devBgImage(bgWidth, bgHeight);
	delete devBgImage.imageData;
	cudaMalloc(&devBgImage.imageData, bgSize);
	cudaMemcpy(devBgImage.getPixels(), bgImage.getPixels(), bgSize, cudaMemcpyHostToDevice);
	
	LpTransformEngine<float> devEngine(patchWidth, patchHeight);
	delete devEngine.lookUpTable;
	cudaMalloc(&devEngine.lookUpTable, engineSize);
	cudaMemcpy(devEngine.lookUpTable, engine.lookUpTable, engineSize, cudaMemcpyHostToDevice);

	MonoImage devPatchImage(patchWidth, patchHeight);
	delete devPatchImage.imageData;
	cudaMalloc(&devPatchImage.imageData, patchSize);
	cudaMemcpy(devPatchImage.getPixels(), patchImage.getPixels(), patchSize, cudaMemcpyHostToDevice);

	MonoImage devTempLpPatch(patchWidth, patchHeight);
	delete devTempLpPatch.imageData;
	cudaMalloc(&devTempLpPatch.imageData, patchSize);
	
	// get LP of patch image
	dim3 dimBlockLP(BLOCK_SIZE_X, BLOCK_SIZE_Y);
	int grid_x = patchWidth % BLOCK_SIZE_X ? patchWidth / dimBlockLP.x + 1 : patchWidth / dimBlockLP.x;
	int grid_y = patchHeight % BLOCK_SIZE_Y ? patchHeight / dimBlockLP.y + 1 : patchHeight / dimBlockLP.y;
	dim3 dimGridLP(grid_x, grid_y);
	//printf("\nGrid (%d, %d), Block (%d, %d)\n", dimGridLP.x, dimGridLP.y, dimBlockLP.x, dimBlockLP.y);
	Simple2dPoint<int> origin = { 0, 0 };
	kernelGetLpImage << < dimGridLP, dimBlockLP >> > (devPatchImage, devLpPatchImage, devEngine, origin);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "kernelGetLpImage launch failed: %s\n", cudaGetErrorString(cudaStatus));

	// search 4D space
	dim3 dimGrid(bgWidth - patchWidth, rowsPerKernel);
	dim3 dimBlock(patchWidth, rowsPerKernel);
	//printf("\nGrid (%d, %d), Block (%d, %d)\n", dimGrid.x, dimGrid.y, dimBlock.x, dimBlock.y);
	Simple2dPoint<int> patchDims = { patchWidth, patchHeight };
	for (int startRow = 0; startRow < bgHeight - patchHeight; startRow += rowsPerKernel) {
		//printf("\r%0.2f%%... ", (float)startRow / (bgHeight - patchHeight) * 100.0);
		kernelSearch4dSpace << < dimGrid, dimBlock, ((patchWidth * (patchHeight + 1)) * 2 + patchWidth * sizeof(float)) >> > (devBgImage, devLpPatchImage, devEngine, startRow, devSearchResults, patchDims, angleLimit);
		cudaThreadSynchronize();
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "\nkernelSearch4dSpace launch failed: %s\n", cudaGetErrorString(cudaStatus));
			break;
		}
	}	

	cudaMemcpy(results, devSearchResults, (bgWidth - patchWidth) * (bgHeight - patchHeight) * sizeof(struct searchResults), cudaMemcpyDeviceToHost);
	
#if SAVE_LP_IMAGE
	cudaMemcpy(patchImage.getPixels(), devLpPatchImage.getPixels(), patchSize, cudaMemcpyDeviceToHost);
#endif

	int bestIndex = 0;
	double bestValue = CORREL_MEASURE ? FLT_MAX : 0.0;
	for (int i = 0; i < (bgWidth - patchWidth) * (bgHeight - patchHeight); i++) {
		Results *res = (results + i);
		if (CORREL_MEASURE ? res->result < bestValue : res->result > bestValue){
			bestIndex = i;
			bestValue = res->result;
		}
	}
	Results *res = (results + bestIndex);
	*sliceResult = *res;
	slicePos->x = bestIndex % (bgWidth - patchWidth); slicePos->y = bestIndex / (bgWidth - patchWidth);

	cudaFree(devBgImage.imageData);
	cudaFree(devLpPatchImage.imageData);
	cudaFree(devEngine.lookUpTable);
	cudaFree(devSearchResults);
}

void timerStart(void) {
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
}

void timerStopAndPrint(std::string str) {
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&GPUmilliseconds, start, stop);
	printf("%s", str.c_str());
	printf("%f", GPUmilliseconds / 1000);
}

unsigned long upperPowOfTwo(unsigned long v) {
	v--;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	v++;
	return v;
}