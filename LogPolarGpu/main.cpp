#include "LogPolar.h"
#include "GpuLogPolar.h"
#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "opencv2/imgproc/imgproc.hpp"
#include <string>
#include <omp.h>

#define LOG_TO_CSV 0

const std::string testDir("../../LogPolarGpu/LogPolarTest/");
const std::string fileExtension(".jpg");
std::string imageDir, refImg, patchImageName;
extern int rowsPerKernel, angleLimit;
int confidenceLevel = 80;

int main(int argc, char *argv[]) {

	if (argc > 3 && argc < 8) {
			patchImageName = argv[1];
			refImg = argv[2];
			imageDir = argv[3];
		if (argc > 4)
			angleLimit = atoi(argv[4]);
		if (argc > 5)
			confidenceLevel = atoi(argv[5]);
		if (argc > 6)
			rowsPerKernel = atoi(argv[6]);
	} else {
		printf("%s: Invalid number of arguments.\nUsage:   LogPolarGpu.exe PATCH_NAME BG_NAME IMAGE_DIR [CONFIDENCE_LEVEL=80] [ANGLE_LIMIT=360] [ROWS_PER_KERNEL=1]\n", argv[0]);
		printf("Example: LogPolarGpu.exe patch1 bg1 oxbuild 80 10\n\n");
		return 1;
	}

	std::string refImgPath(testDir + imageDir + "/images/" + refImg + fileExtension);
	std::string patchImgPath(testDir + imageDir + "/patterns/" + patchImageName + fileExtension);

	cv::Mat refImageCV, patchImageCV;		
	refImageCV = cv::imread(refImgPath.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
	patchImageCV = cv::imread(patchImgPath.c_str(), CV_LOAD_IMAGE_GRAYSCALE);

	if (!refImageCV.data) {
		printf("No ref image data at %s\n", refImgPath.c_str());
		return 1;
	}
	if (!patchImageCV.data) {
		printf("No patch image data at %s\n", patchImgPath.c_str());
		return 1;
	}

	int refWidth = refImageCV.cols;
	int refHeight = refImageCV.rows;
	int patchWidth = patchImageCV.cols;
	int patchHeight = patchImageCV.rows;


	/*
	double alpha = 1.0; int beta = 10;
	for (int y = 0; y < refImageCV.rows; y++)
		for (int x = 0; x < refImageCV.cols; x++)
				refImageCV.at<int>(cv::Point(x, y)) = cv::saturate_cast<uchar>(alpha*(refImageCV.at<int>(cv::Point(x, y))) + beta);
	for (int y = 0; y < patchImageCV.rows; y++) 
		for (int x = 0; x < patchImageCV.cols; x++) 
				patchImageCV.at<int>(cv::Point(x, y)) = cv::saturate_cast<uchar>(alpha*(patchImageCV.at<int>(cv::Point(x, y))) + beta);
	*/

	/*
	cv::cvtColor(refImageCV, refImageCV, CV_BGR2YUV);
	std::vector<cv::Mat> channels;
	cv::split(refImageCV, channels);
	cv::equalizeHist(channels[0], channels[0]);
	cv::merge(channels, refImageCV);
	cv::cvtColor(refImageCV, refImageCV, CV_YUV2BGR);
	cv::cvtColor(refImageCV, refImageCV, cv::COLOR_RGB2GRAY);

	cv::cvtColor(patchImageCV, patchImageCV, CV_BGR2YUV);
	//std::vector<cv::Mat> channels;
	cv::split(patchImageCV, channels);
	cv::equalizeHist(channels[0], channels[0]);
	cv::merge(channels, patchImageCV);
	cv::cvtColor(patchImageCV, patchImageCV, CV_YUV2BGR);
	cv::cvtColor(patchImageCV, patchImageCV, cv::COLOR_RGB2GRAY);
	*/

	/*
	threshold(refImageCV, refImageCV, 63, 255, CV_THRESH_TOZERO);
	threshold(patchImageCV, patchImageCV, 15, 255, CV_THRESH_TOZERO);
	*/

	size_t patchDataSize = patchWidth * patchHeight;
	MonoImage patchImage(patchWidth, patchHeight);

	memcpy((void*)patchImage.getPixels(), patchImageCV.data, patchDataSize);

	Simple2dPoint<float> transformCenter = { patchWidth / 2.0f, patchHeight / 2.0f };
	bool useDiagonalAsRadius = false;
	LpTransformEngine<float> engine = GetLpTransformEngine(patchWidth, patchHeight, transformCenter, useDiagonalAsRadius);

	int gpuCount = 0;
	cudaGetDeviceCount(&gpuCount);

	if (gpuCount < 1) {
		printf("no CUDA capable devices were detected\n");
		return 1;
	}

	int refSliceHeight = ((refHeight - patchHeight) % gpuCount == 0) ? ((refHeight - patchHeight) / gpuCount) : ((refHeight - patchHeight) / gpuCount + 1);
	Results *results = new Results[gpuCount];
	Simple2dPoint<int> *slicePos = new Simple2dPoint<int>[gpuCount];

#if LOG_TO_CSV
	printf("%s;%s;%s;", imageDir.c_str(), refImg.c_str(), patchImageName.c_str());
#else
	printf("%s, Slice height: %d", (imageDir + ", " + refImg + ", " + patchImageName).c_str(), refSliceHeight);
#endif

	omp_set_num_threads(gpuCount);  // create as many CPU threads as there are CUDA devices
	timerStart();

#pragma omp parallel
	{
		unsigned int cpuTID = omp_get_thread_num();
		size_t refDataSize = (refSliceHeight + patchHeight) * refWidth;
		MonoImage bgImage(refWidth, refSliceHeight + patchHeight);
		memcpy((void*)(bgImage.getPixels()), refImageCV.data + cpuTID * refSliceHeight * refWidth, refDataSize);
		// set and check the CUDA device for this CPU thread
		int gpuID = -1;
		cudaSetDevice(cpuTID);
		gpuGetLpImage(bgImage, patchImage, engine, results + cpuTID, slicePos + cpuTID);
	}
#if LOG_TO_CSV
	timerStopAndPrint("");
#else
	timerStopAndPrint("\nLogPolarGpuTime: ");
#endif

#if SAVE_LP_IMAGE
	memcpy(patchImageCV.data, (void*)patchImage.getPixels(), patchDataSize);
	cv::imwrite(testDir + "/outputLPbg" + fileExtension, patchImageCV);
#endif

	if (cudaSuccess != cudaGetLastError())
		printf("%s\n", cudaGetErrorString(cudaGetLastError()));

	int bestIndex = 0;
	// Select best score accross multiple GPUs if used
	double bestValue = CORREL_MEASURE ? FLT_MAX : 0.0;
	for (int i = 0; i < gpuCount; i++) {
		Results *res = (results + i);
		if (CORREL_MEASURE ? res->result < bestValue : res->result > bestValue) {
			bestIndex = i;
			bestValue = res->result;
		}
	}

	int found_x = slicePos[bestIndex].x, found_y = slicePos[bestIndex].y + bestIndex * refSliceHeight;
	float angle = results[bestIndex].angle == 0 ? 0.0f : (360.0f - (float)results[bestIndex].angle / (float)patchHeight * 360.0f);
	Simple2dPoint<float> centerPoint = { (float)patchWidth / 2.0f, (float)patchHeight / 2.0f };
	double logBase = GetLogBaseToCoverTheSpace<float>(centerPoint, patchWidth, patchHeight, patchWidth, false);
	float cartesianScale = powf(logBase, (float)(results[bestIndex].scale - patchWidth / 2)) * 100.0f;

#if LOG_TO_CSV
	printf(";%d;%d;%0.2f;%d;%02f;%d;%0.6f;\n", found_x, found_y, angle, results[bestIndex].angle % patchHeight, cartesianScale, results[bestIndex].scale, results[bestIndex].result);
#else  
	printf(" Position: (%d, %d), ", found_x, found_y);
	printf("angle: %0.1f deg (%d/%dpx), ", angle, results[bestIndex].angle % patchHeight, patchHeight);
	printf("scale: %0.1f%% (%d/%dpx), ", cartesianScale, results[bestIndex].scale, patchWidth);
	printf("result: %0.3f\n\n", results[bestIndex].result);
#endif

	// Do not save result image if correlation factor is less than 0.8 for CovVar or more than 2000 for SAD and CENSUS
	if (CORREL_MEASURE ? results[bestIndex].result > 2000.0 : results[bestIndex].result < (float)confidenceLevel / 100.0) return 1;

	// Open original Reference Image and draw rectangle using previously calculated parameters
	cv::Mat outputImage = cv::imread(testDir + imageDir + "/images/" + refImg + fileExtension, CV_LOAD_IMAGE_COLOR);
	cv::Point2f center(found_x + patchWidth / 2, found_y + patchHeight / 2);
	cv::Size2f size((float)(patchWidth) * (cartesianScale / 100.0f), (float)(patchHeight) * (cartesianScale / 100.0f));
	cv::RotatedRect rect(center, size, angle);
	cv::Point2f vertices[4];
	rect.points(vertices);
	for (int i = 0; i < 4; i++)
		line(outputImage, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 0, 255), (refWidth / 160) > 0 ? refWidth / 160 : 1);
	std::ostringstream conf_lev;
	conf_lev << results[bestIndex].result;
	// Save results in new file
	cv::imwrite(testDir + imageDir + "/results/" + conf_lev.str() + "_" + refImg + fileExtension, outputImage);

	return 0;
}