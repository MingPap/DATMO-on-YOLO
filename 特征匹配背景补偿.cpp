#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/features2d.hpp"

#include <iostream>
#include <fstream>
#include<string>
#include<opencv2/optflow.hpp>
#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/cudaoptflow.hpp"
#include "opencv2/cudaarithm.hpp"


using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

const int MAX_FEATURES = 500;
const float GOOD_MATCH_PERCENT = 0.15f;

inline bool isFlowCorrect(Point2f u)
{
	return !cvIsNaN(u.x) && !cvIsNaN(u.y) && fabs(u.x) < 1e9 && fabs(u.y) < 1e9;
}

static Vec3b computeColor(float fx, float fy)
{
	static bool first = true;

	// relative lengths of color transitions:
	// these are chosen based on perceptual similarity
	// (e.g. one can distinguish more shades between red and yellow
	//  than between yellow and green)
	const int RY = 15;
	const int YG = 6;
	const int GC = 4;
	const int CB = 11;
	const int BM = 13;
	const int MR = 6;
	const int NCOLS = RY + YG + GC + CB + BM + MR;
	static Vec3i colorWheel[NCOLS];

	if (first)
	{
		int k = 0;

		for (int i = 0; i < RY; ++i, ++k)
			colorWheel[k] = Vec3i(255, 255 * i / RY, 0);

		for (int i = 0; i < YG; ++i, ++k)
			colorWheel[k] = Vec3i(255 - 255 * i / YG, 255, 0);

		for (int i = 0; i < GC; ++i, ++k)
			colorWheel[k] = Vec3i(0, 255, 255 * i / GC);

		for (int i = 0; i < CB; ++i, ++k)
			colorWheel[k] = Vec3i(0, 255 - 255 * i / CB, 255);

		for (int i = 0; i < BM; ++i, ++k)
			colorWheel[k] = Vec3i(255 * i / BM, 0, 255);

		for (int i = 0; i < MR; ++i, ++k)
			colorWheel[k] = Vec3i(255, 0, 255 - 255 * i / MR);

		first = false;
	}

	const float rad = sqrt(fx * fx + fy * fy);
	const float a = atan2(-fy, -fx) / (float)CV_PI;

	const float fk = (a + 1.0f) / 2.0f * (NCOLS - 1);
	const int k0 = static_cast<int>(fk);
	const int k1 = (k0 + 1) % NCOLS;
	const float f = fk - k0;

	Vec3b pix;

	for (int b = 0; b < 3; b++)
	{
		const float col0 = colorWheel[k0][b] / 255.0f;
		const float col1 = colorWheel[k1][b] / 255.0f;

		float col = (1 - f) * col0 + f * col1;

		if (rad <= 1)
			col = 1 - rad * (1 - col); // increase saturation with radius
		else
			col *= .75; // out of range

		pix[2 - b] = static_cast<uchar>(255.0 * col);
	}

	return pix;
}

static void drawOpticalFlow(const Mat_<float>& flowx, const Mat_<float>& flowy, Mat& dst, float maxmotion = -1)
{
	dst.create(flowx.size(), CV_8UC3);
	dst.setTo(Scalar::all(0));

	// determine motion range:
	float maxrad = maxmotion;

	if (maxmotion <= 0)
	{
		maxrad = 1;
		for (int y = 0; y < flowx.rows; ++y)
		{
			for (int x = 0; x < flowx.cols; ++x)
			{
				Point2f u(flowx(y, x), flowy(y, x));

				if (!isFlowCorrect(u))
					continue;

				maxrad = max(maxrad, sqrt(u.x * u.x + u.y * u.y));
			}
		}
	}

	for (int y = 0; y < flowx.rows; ++y)
	{
		for (int x = 0; x < flowx.cols; ++x)
		{
			Point2f u(flowx(y, x), flowy(y, x));

			if (isFlowCorrect(u))
				dst.at<Vec3b>(y, x) = computeColor(u.x / maxrad, u.y / maxrad);
		}
	}
}

Mat optic(Mat pre, Mat aft)
{
	//cv::cvtColor(pre, pre, COLOR_BGR2GRAY);
	//cv::cvtColor(aft, aft, COLOR_BGR2GRAY);

	cuda::GpuMat d_frame0(pre);
	cuda::GpuMat d_frame1(aft);

	cuda::GpuMat d_flow(pre.size(), CV_32FC2);

	//Ptr<cuda::FarnebackOpticalFlow> farn = cuda::FarnebackOpticalFlow::create();

	Ptr<cuda::BroxOpticalFlow> brox = cuda::BroxOpticalFlow::create(0.197f, 50.0f, 0.8f, 10, 77, 10);
	cuda::GpuMat d_frame0f;
	cuda::GpuMat d_frame1f;

	d_frame0.convertTo(d_frame0f, CV_32F, 1.0 / 255.0);
	d_frame1.convertTo(d_frame1f, CV_32F, 1.0 / 255.0);


	const int64 start = getTickCount();

	//farn->calc(d_frame0, d_frame1, d_flow);
	brox->calc(d_frame0f, d_frame1f, d_flow);

	const double timeSec = (getTickCount() - start) / getTickFrequency();
	//cout << "farn : " << timeSec << " sec" << endl;
	cout << "Brox : " << timeSec << " sec" << endl;

	cuda::GpuMat planes[2];
	cuda::split(d_flow, planes);

	Mat flowx(planes[0]);
	Mat flowy(planes[1]);

	Mat out, mag, ang;

	drawOpticalFlow(flowx, flowy, out);

	cv::cartToPolar(flowx, flowy, mag, ang, true);
	//normalize(mag, mag, 0, 1, NORM_MINMAX);

	//光流梯度分割
	float n1 = 0.1;
	Mat sm = mag.clone();
	for (int i = 0; i < sm.rows; ++i)
	{
		float *p1 = sm.ptr<float>(i);
		float *p2 = mag.ptr<float>(i);
		for (int j = 0; j < sm.cols; ++j)
		{
			float t = exp(p2[j] * n1);
			p1[j] = 1 - 1 / t;
		}
	}

	////光流矢量方向分割
	float n2 = 0.1;
	Mat sg = ang.clone();
	//maxfilter
	Mat element = Mat::ones(3, 3, ang.type());
	dilate(ang, sg, element);

	for (int i = 0; i < sg.rows; ++i)
	{
		float *p = sg.ptr<float>(i);
		for (int j = 0; j < sg.cols; ++j)
		{
			p[j] = 1 - 1 / (n2*exp(p[j]));
		}
	}


	float N = 0.6;
	Mat mask = Mat::zeros(sg.rows, sg.cols, CV_8U);

	for (int i = 0; i < sg.rows; ++i)
	{
		float *m = sm.ptr<float>(i);
		float *a = sg.ptr<float>(i);
		uchar *t = mask.ptr<uchar>(i);
		for (int j = 0; j < sg.cols; ++j)
		{
			if (m[j] * a[j] >= N)
				t[j] = 255;
		}
	}

	return mask;
}



void alignImages(Mat &im1, Mat &im2, Mat &im1Reg, Mat &h)

{

	// Convert images to grayscale
	Mat im1Gray, im2Gray;
	cvtColor(im1, im1Gray, CV_BGR2GRAY);
	cvtColor(im2, im2Gray, CV_BGR2GRAY);

	// Variables to store keypoints and descriptors
	std::vector<KeyPoint> keypoints1, keypoints2;
	Mat descriptors1, descriptors2;

	//// Detect ORB features and compute descriptors.
	//Ptr<Feature2D> orb = ORB::create(MAX_FEATURES);
	//orb->detectAndCompute(im1Gray, Mat(), keypoints1, descriptors1);
	//orb->detectAndCompute(im2Gray, Mat(), keypoints2, descriptors2);

	//Surf特征点
	Ptr<SurfFeatureDetector> surf = SURF::create(MAX_FEATURES);
	surf->detectAndCompute(im1Gray, Mat(), keypoints1, descriptors1);
	surf->detectAndCompute(im2Gray, Mat(), keypoints2, descriptors2);

	// Match features.
	std::vector<DMatch> matches;
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(3);         //BRUTEFORCE_L1匹配
	matcher->match(descriptors1, descriptors2, matches, Mat());          



	// Sort matches by score
	std::sort(matches.begin(), matches.end());

	// Remove not so good matches
	const int numGoodMatches = matches.size() * GOOD_MATCH_PERCENT;
	matches.erase(matches.begin() + numGoodMatches, matches.end());


	// Draw top matches
	Mat imMatches;
	drawMatches(im1, keypoints1, im2, keypoints2, matches, imMatches);
	//imwrite("matches.jpg", imMatches);
	imshow("matches.jpg", imMatches);


	// Extract location of good matches
	std::vector<Point2f> points1, points2;

	for (size_t i = 0; i < matches.size(); i++)
	{
		points1.push_back(keypoints1[matches[i].queryIdx].pt);
		points2.push_back(keypoints2[matches[i].trainIdx].pt);
	}


	//仿射变换

	Mat warpMat = estimateRigidTransform(points1, points2, true);

	warpAffine(im1, im1Reg, warpMat, im1.size());

	//// Find homography
	//h = findHomography(points1, points2, RANSAC);

	//// Use homography to warp image
	//warpPerspective(im1, im1Reg, h, im2.size());

}


int main(int argc, char **argv)
{
	// Read reference image
	string refFilename("E:/MyFiles/desktop/testimage/45/000045_11.png");
	cout << "Reading reference image : " << refFilename << endl;
	Mat imReference = imread(refFilename);


	// Read image to be aligned
	string imFilename("E:/MyFiles/desktop/testimage/45/000045_10.png");
	cout << "Reading image to align : " << imFilename << endl;
	Mat im = imread(imFilename);


	// Registered image will be resotred in imReg. 
	// The estimated homography will be stored in h. 
	Mat imReg, h;

	// Align images
	cout << "Aligning images ..." << endl;
	alignImages(im, imReference, imReg, h);
	cvtColor(imReg, imReg, CV_BGR2GRAY);
	cvtColor(imReference, imReference, CV_BGR2GRAY);

	Mat mask;
	cv::threshold(imReg, mask, 0, 1, THRESH_BINARY);
	Mat imRe = imReference.mul(mask);


	// Write aligned image to disk. 
	string outFilename("aligned.jpg");
	cout << "Saving aligned image : " << outFilename << endl;
	imshow(outFilename, imReg);
	imshow("imReference", imReference);
	Mat dif1,dif2;
	////imwrite(outFilename, imReg);
	absdiff(imReg, imRe, dif1);
	//absdiff(im, imReference, dif2);

	Mat of = optic(imReg, imRe);

	//cvtColor(dif1, d1_gray, CV_BGR2GRAY);
	//cvtColor(dif2, d2_gray, CV_BGR2GRAY);
	
	
	//寻找图像中的最大最小值             二值化
	double min, max;
	minMaxLoc(dif1, &min, &max, NULL, NULL);
	double k = 0.5;
	Mat th;
	cv::threshold(dif1, th, k*(max-min), 255, THRESH_BINARY);

	//开运算
	Mat result;
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(4, 4));
	morphologyEx(th, result, MORPH_OPEN, element);

	// Print estimated homography
	cout << "Estimated homography : \n" << h << endl;
	waitKey();
	
	system("pause");
}