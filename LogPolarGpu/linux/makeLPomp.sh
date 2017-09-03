nvcc -Xcompiler \-fopenmp -lgomp ../LogPolar.cu ../main.cpp ../LogPolar.cpp -gencode arch=compute_20,code=sm_20 -o ../../linux/Release/LogPolarGpu -L/software/local/libs/OpenCV/2.4.2/lib -lopencv_imgproc -lopencv_highgui -lopencv_core -Wno-deprecated-gpu-targets -I/software/local/libs/OpenCV/2.4.2/include

