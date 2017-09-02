#include "LogPolar.h"

template LpTransformEngine<double> GetLpTransformEngine(const int cols, const int rows, const Simple2dPoint<double> & centralPoint, bool diagonal);
template LpTransformEngine<float> GetLpTransformEngine(const int cols, const int rows, const Simple2dPoint<float> & centralPoint, bool diagonal);

template <class T>
LpTransformEngine<T> GetLpTransformEngine(const int cols, const int rows, const Simple2dPoint<T> & centralPoint, bool diagonal) {
	
	LpTransformEngine<T> engine(cols, rows);

	// In the log-polar image the horizontal axis denotes "r", whereas the vertical "phi".
	double r_max = cols;
	double theLnBase = GetLogBaseToCoverTheSpace<T>(centralPoint, cols, rows, r_max, diagonal);
	double * power_LUT = new double[cols];

	for (int i = 0; i < cols; i++) 		// B^r
		power_LUT[i] = pow(theLnBase, (double)i);

	const double kFromAngle = 0, kToAngle = k2Pi;
	const double kStep = (kToAngle - kFromAngle) / (double)rows;

	double c1 = centralPoint.x, c2 = centralPoint.y;
	double x1, x2, alpha, cos_alpha, sin_alpha, power_value;

	// Go through angles (rows)
	int i;
	for (alpha = kFromAngle, i = 0; i < rows; alpha += kStep, i++) {
		cos_alpha = cos(alpha);
		sin_alpha = sin(alpha);
		// Go through radius (columns)
		for (int j = 0; j < cols; ++j) {
			power_value = power_LUT[j];
			x1 = cos_alpha * power_value + c1;
			x2 = sin_alpha * power_value + c2;
			if (diagonal == false) {
				assert(x1 < (double)cols && x1 > 0.0);
				assert(x2 < (double)rows && x2 > 0.0);
			}
			Simple2dPoint<T> point = { (T)x1, (T)x2 };
			engine.setTransformCords(j, i, point);
		}
	}

	delete[] power_LUT;

	return engine;
}

template <class T>
double GetLogBaseToCoverTheSpace(const Simple2dPoint<T> & point, int cols, int rows, double r_max, bool diagonal) {
	double dx_max, dy_max, d_max, B;
	if (diagonal == true) {
		dx_max = max(point.x, cols - point.x);
		dy_max = max(point.y, rows - point.y);
		d_max = dx_max * dx_max + dy_max * dy_max;
		B = exp(log(d_max) / (2.0 * r_max));
	}
	else {
		dx_max = min(point.x, cols - point.x);
		dy_max = min(point.y, rows - point.y);
		d_max = min(dx_max, dy_max);
		B = exp(log(d_max) / r_max);
	}
	return B;
}
