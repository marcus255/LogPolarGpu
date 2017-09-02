#ifndef GPU_LOG_POLAR_H
#define GPU_LOG_POLAR_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "string"
#include <stdio.h>
#include "LogPolar.h"

#define COV_VAR 0
#define SAD     1
#define CENSUS  2

#define CORREL_MEASURE COV_VAR
#define SAVE_LP_IMAGE 0

typedef struct searchResults { int angle; int scale; double result; } Results;

// Function prototypes
__global__ void kernelGetLpImage(const MonoImage bgImage, MonoImage patchImage, const LpTransformEngine<float> engine, Simple2dPoint<int> patchPosition);
__global__ void kernelSearch4dSpace(const MonoImage devBgImage, const MonoImage devLpPatchImage, const LpTransformEngine<float> devEngine, int startRow, Results *devSearchResults, Simple2dPoint<int> patchSize, int angLim = 360);
void gpuGetLpImage(const MonoImage & bgImage, const MonoImage & lpPatchImage, const LpTransformEngine<float> & engine, Results *sliceResult, Simple2dPoint<int> *slicePos);
void timerStart(void);
void timerStopAndPrint(std::string str = 0);
unsigned long upperPowOfTwo(unsigned long v);
CUDA_HOSTDEV uint8_t interpolate(const MonoImage devBgImage, Simple2dPoint<float> point, Simple2dPoint<int> position);
CUDA_HOSTDEV float getMeanValue(uint8_t *buffer, int cols, int rows, int commonCols, int offset);
CUDA_HOSTDEV float computeCorelation(const uint8_t *buf_1, const uint8_t *buf_2, int cols, int rows, float mean_1, float mean_2, int angleShift, int commonCols, int bgOffset, int patchOffset);
CUDA_HOSTDEV float computeSADScore(const uint8_t *lpBgPatchBuf, const uint8_t *lpPatchBuf, int cols, int rows, int angleShift, int commonCols, int bgOffset, int patchOffset);
__device__ float computeCensusScore(const uint8_t *lpBgPatchBuf, const uint8_t *lpPatchBuf, int cols, int rows, int angleShift, int commonCols, int bgOffset, int patchOffset);

#endif // GPU_LOG_POLAR_H
