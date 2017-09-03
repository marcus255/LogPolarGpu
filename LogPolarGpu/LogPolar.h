#ifndef LOG_POLAR_H
#define LOG_POLAR_H

#include <math.h>
#include <assert.h>
#include <stdio.h>

typedef unsigned char uint8_t;

#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

#define k2Pi (2.0 * 3.1415926535897932384626433832795)

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

template <class T>
struct Simple2dPoint {
	T x;
	T y;
};

class MonoImage {
private:
	int cols;
	int rows;
	int pixels;
public:
	uint8_t *imageData;
public:
	CUDA_HOSTDEV MonoImage() { imageData = 0; }
	CUDA_HOSTDEV MonoImage(int width, int height) {
		cols = width; rows = height;
		pixels = cols * rows;
		imageData = new uint8_t[pixels];
	}
	CUDA_HOSTDEV inline uint8_t getPixel(int col, int row) const { return imageData[row * cols + col]; }
	CUDA_HOSTDEV inline void setPixel(int col, int row, uint8_t value) { imageData[row * cols + col] = value; }
	CUDA_HOSTDEV inline int getWidth(void) const { return cols; }
	CUDA_HOSTDEV inline int getHeight(void) const { return rows; }
	CUDA_HOSTDEV inline void setPixels(uint8_t* data) { imageData = data; }
	CUDA_HOSTDEV inline uint8_t* getPixels(void) const { return imageData; }
};

template <class T>
class LpTransformEngine {
private:
	int cols;
	int rows;
	int tableSize;
public:
	Simple2dPoint<T> *lookUpTable;
public:
	CUDA_HOSTDEV LpTransformEngine() { lookUpTable = 0; }
	CUDA_HOSTDEV LpTransformEngine(int width, int height) {
		cols = width; rows = height;
		tableSize = cols * rows;
		lookUpTable = new Simple2dPoint<T>[tableSize];
	}
	CUDA_HOSTDEV inline int getTransformOffset(int position) const { return lookUpTable[position]; }
	CUDA_HOSTDEV inline void setTransformOffset(int position, int offset) { lookUpTable[position] = offset; }
	CUDA_HOSTDEV inline Simple2dPoint<T> getTransformCords(int x, int y) const { return lookUpTable[y * cols + x]; }
	CUDA_HOSTDEV inline void setTransformCords(int x, int y, Simple2dPoint<T> point) { lookUpTable[y * cols + x] = point; }
};

template <class T> LpTransformEngine<T> GetLpTransformEngine(const int cols, const int rows, const Simple2dPoint<T> & centralPoint, bool diagonal);
template <class T> double GetLogBaseToCoverTheSpace(const Simple2dPoint<T> & anchorPoint, int cols, int rows, double r_max, bool diagonal);

#endif // LOG_POLAR_H