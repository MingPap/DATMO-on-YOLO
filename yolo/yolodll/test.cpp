#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <queue>
#include <fstream>
#include <thread>
#include <future>
#include <atomic>
#include <mutex>         // std::mutex, std::unique_lock
#include <cmath>
#include"func.h"

#include<opencv2/optflow.hpp>



// It makes sense only for video-Camera (not for video-File)
// To use - uncomment the following line. Optical-flow is supported only by OpenCV 3.x - 4.x
//#define TRACK_OPTFLOW
#define OPENCV
#define GPU

// To use 3D-stereo camera ZED - uncomment the following line. ZED_SDK should be installed.
//#define ZED_STEREO


#include "yolo_v2_class.hpp"    // imported functions from DLL

#ifdef OPENCV
#ifdef ZED_STEREO
#include <sl_zed/Camera.hpp>
#pragma comment(lib, "sl_core64.lib")
#pragma comment(lib, "sl_input64.lib")
#pragma comment(lib, "sl_zed64.lib")

float getMedian(std::vector<float> &v) {
	size_t n = v.size() / 2;
	std::nth_element(v.begin(), v.begin() + n, v.end());
	return v[n];
}

std::vector<bbox_t> get_3d_coordinates(std::vector<bbox_t> bbox_vect, cv::Mat xyzrgba)
{
	bool valid_measure;
	int i, j;
	const unsigned int R_max_global = 10;

	std::vector<bbox_t> bbox3d_vect;

	for (auto &cur_box : bbox_vect) {

		const unsigned int obj_size = std::min(cur_box.w, cur_box.h);
		const unsigned int R_max = std::min(R_max_global, obj_size / 2);
		int center_i = cur_box.x + cur_box.w * 0.5f, center_j = cur_box.y + cur_box.h * 0.5f;

		std::vector<float> x_vect, y_vect, z_vect;
		for (int R = 0; R < R_max; R++) {
			for (int y = -R; y <= R; y++) {
				for (int x = -R; x <= R; x++) {
					i = center_i + x;
					j = center_j + y;
					sl::float4 out(NAN, NAN, NAN, NAN);
					if (i >= 0 && i < xyzrgba.cols && j >= 0 && j < xyzrgba.rows) {
						cv::Vec4f &elem = xyzrgba.at<cv::Vec4f>(j, i);  // x,y,z,w
						out.x = elem[0];
						out.y = elem[1];
						out.z = elem[2];
						out.w = elem[3];
					}
					valid_measure = std::isfinite(out.z);
					if (valid_measure)
					{
						x_vect.push_back(out.x);
						y_vect.push_back(out.y);
						z_vect.push_back(out.z);
					}
				}
			}
		}

		if (x_vect.size() * y_vect.size() * z_vect.size() > 0)
		{
			cur_box.x_3d = getMedian(x_vect);
			cur_box.y_3d = getMedian(y_vect);
			cur_box.z_3d = getMedian(z_vect);
		}
		else {
			cur_box.x_3d = NAN;
			cur_box.y_3d = NAN;
			cur_box.z_3d = NAN;
		}

		bbox3d_vect.emplace_back(cur_box);
	}

	return bbox3d_vect;
}

cv::Mat slMat2cvMat(sl::Mat &input) {
	// Mapping between MAT_TYPE and CV_TYPE
	int cv_type = -1;
	switch (input.getDataType()) {
	case sl::MAT_TYPE_32F_C1:
		cv_type = CV_32FC1;
		break;
	case sl::MAT_TYPE_32F_C2:
		cv_type = CV_32FC2;
		break;
	case sl::MAT_TYPE_32F_C3:
		cv_type = CV_32FC3;
		break;
	case sl::MAT_TYPE_32F_C4:
		cv_type = CV_32FC4;
		break;
	case sl::MAT_TYPE_8U_C1:
		cv_type = CV_8UC1;
		break;
	case sl::MAT_TYPE_8U_C2:
		cv_type = CV_8UC2;
		break;
	case sl::MAT_TYPE_8U_C3:
		cv_type = CV_8UC3;
		break;
	case sl::MAT_TYPE_8U_C4:
		cv_type = CV_8UC4;
		break;
	default:
		break;
	}
	return cv::Mat(input.getHeight(), input.getWidth(), cv_type, input.getPtr<sl::uchar1>(sl::MEM_CPU));
}

cv::Mat zed_capture_rgb(sl::Camera &zed) {
	sl::Mat left;
	zed.retrieveImage(left);
	return slMat2cvMat(left).clone();
}

cv::Mat zed_capture_3d(sl::Camera &zed) {
	sl::Mat cur_cloud;
	zed.retrieveMeasure(cur_cloud, sl::MEASURE_XYZ);
	return slMat2cvMat(cur_cloud).clone();
}

static sl::Camera zed; // ZED-camera

#else   // ZED_STEREO


std::vector<bbox_t> get_3d_coordinates(std::vector<bbox_t> bbox_vect, cv::Mat xyzrgba) {
	return bbox_vect;
}
#endif  // ZED_STEREO


#include <opencv2/opencv.hpp>            // C++
#include <opencv2/core/version.hpp>
#ifndef CV_VERSION_EPOCH     // OpenCV 3.x and 4.x
#include <opencv2/videoio/videoio.hpp>
#define OPENCV_VERSION CVAUX_STR(CV_VERSION_MAJOR)"" CVAUX_STR(CV_VERSION_MINOR)"" CVAUX_STR(CV_VERSION_REVISION)
#ifndef USE_CMAKE_LIBS
#pragma comment(lib, "opencv_world" OPENCV_VERSION ".lib")
#ifdef TRACK_OPTFLOW
#pragma comment(lib, "opencv_cudaoptflow" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_cudaimgproc" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_core" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_imgproc" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_highgui" OPENCV_VERSION ".lib")
#endif    // TRACK_OPTFLOW
#endif    // USE_CMAKE_LIBS
#else     // OpenCV 2.x
#define OPENCV_VERSION CVAUX_STR(CV_VERSION_EPOCH)"" CVAUX_STR(CV_VERSION_MAJOR)"" CVAUX_STR(CV_VERSION_MINOR)
#ifndef USE_CMAKE_LIBS
#pragma comment(lib, "opencv_core" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_imgproc" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_highgui" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_video" OPENCV_VERSION ".lib")
#endif    // USE_CMAKE_LIBS
#endif    // CV_VERSION_EPOCH

//std::vector<bbox_t>  result_confus(std::vector<bbox_t> vec1, std::vector<bbox_t> vec2)
//{
//	std::vector<bbox_t> TEMP=vec1;
//	for (int i = 0; i < vec1.size(); ++i)
//	{
//		//for (int j = 0; j < vec2.size(); ++j)
//		//{
//		//	double t = abs(vec1[i].x - vec2[j].x) + abs(vec1[i].y - vec2[j].y);
//		//	double IOU = 0.0;
//		//	if (t <1000.0)
//		//	{
//				//Rect rect;
//				//rect.x = vec1[i].x;
//				//rect.y = vec1[i].y;
//				//rect.width = vec1[i].w;
//				//rect.height = vec1[i].h;
//
//				//Rect rect1;
//				//rect1.x = vec2[i].x;
//				//rect1.y = vec2[i].y;
//				//rect1.width = vec2[i].w;
//				//rect1.height = vec2[i].h;
//				//Rect rect2 = rect | rect1;
//				//Rect rect3 = rect & rect1;
//				//double IOU= rect3.area() *1.0 / rect2.area();
//				//cout << i<<' ' <<IOU << endl;
//
//				int x = (vec1[i].x > vec2[i].x) ? vec2[i].x : vec1[i].x;
//				int y = (vec1[i].y > vec2[i].y) ? vec2[i].y : vec1[i].y;
//				int h = (vec1[i].h > vec2[i].h) ? vec1[i].h : vec2[i].h;
//				int w = (vec1[i].w > vec2[i].w) ? vec1[i].w : vec2[i].w;
//				
//		//	}
//
//		//	//TEMP.push_back()
//		//}
//	}
//	return TEMP;
//}


//void ROI_cut(cv::Mat of_in, std::vector<bbox_t> result_vec_after)
//{
//	int k = 0;
//	for (auto &i : result_vec_after)
//	{
//		cv::Mat image_roi = of_in(cv::Rect(i.x, i.y, i.w, i.h));
//		string t = std::to_string(k);
//		cv::imwrite("E:/MyFiles/desktop/optic/out" + t + ".png", image_roi);
//
//		++k;
//	}
//
//}
//

void alignImages(Mat &im1, Mat &im2, Mat &im1Reg, Mat &h, std::vector<bbox_t> resultVec1, std::vector<bbox_t> resultVec2)

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

	/*for (size_t j = 0; j < keypoints1.size(); j++)
	{
		for (auto &i : resultVec1)
		{

			if (((keypoints1[j].pt.x > i.x && keypoints1[j].pt.x <(i.x + i.w)
				&& keypoints1[j].pt.y > i.y && keypoints1[j].pt.y < (i.y + i.h))))
				keypoints1.erase(keypoints1.begin() + j);

		}
	}
	for (size_t j = 0; j < keypoints1.size(); j++)
	{
		for (auto &i : resultVec2)
		{
			if (((keypoints2[j].pt.x > i.x && keypoints2[j].pt.x <(i.x + i.w)
				&& keypoints2[j].pt.y > i.y && keypoints2[j].pt.y < (i.y + i.h))))
				keypoints2.erase(keypoints2.begin() + j);

		}
	}

	Mat ou1, ou2;
	drawKeypoints(im1, keypoints1, ou1, Scalar(0, 0, 255));
	drawKeypoints(im2, keypoints2, ou2, Scalar(0, 0, 255));*/
	// Match features.
	std::vector<DMatch> matches;
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(3);         //BRUTEFORCE_SL2匹配
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
	//imshow("matches.jpg", imMatches);


	// Extract location of good matches
	std::vector<Point2f> points1, points2;

	for (size_t j = 0; j < matches.size(); j++)
	{
		for (auto &i : resultVec1)
		{
			auto iter = keypoints1[matches[j].queryIdx];
			if ((iter.pt.x > i.x && iter.pt.x <(i.x + i.w)
				&& iter.pt.y > i.y && iter.pt.y < (i.y + i.h)))
				points1.push_back(iter.pt);

		}
		for (auto &i : resultVec2)
		{
			auto iter = keypoints2[matches[j].trainIdx];
			if ((iter.pt.x > i.x && iter.pt.x <(i.x + i.w)
				&& iter.pt.y > i.y && iter.pt.y < (i.y + i.h)))
				points2.push_back(iter.pt);

		}
	}

	//for (size_t j = 0; j < matches.size(); j++)
	//{
	//	points1.push_back(keypoints1[matches[j].queryIdx].pt);
	//	points2.push_back(keypoints2[matches[j].trainIdx].pt);

	//}


	//仿射变换

	Mat warpMat = estimateRigidTransform(points1, points2, true);

	warpAffine(im1, im1Reg, warpMat, im1.size());

}

void draw_boxes(cv::Mat mat_img, std::vector<bbox_t> result_vec, std::vector<std::string> obj_names,
	int current_det_fps = -1, int current_cap_fps = -1)
{
	int const colors[6][3] = { { 1,0,1 },{ 0,0,1 },{ 0,1,1 },{ 0,1,0 },{ 1,1,0 },{ 1,0,0 } };

	for (auto &i : result_vec) {
		cv::Scalar color = obj_id_to_color(i.obj_id);
		cv::rectangle(mat_img, cv::Rect(i.x, i.y, i.w, i.h), color, 2);   //在图片上绘制矩形
		if (obj_names.size() > i.obj_id) {
			std::string obj_name = obj_names[i.obj_id];
			if (i.track_id > 0) obj_name += " - " + std::to_string(i.track_id);
			cv::Size const text_size = getTextSize(obj_name, cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, 2, 0);
			int max_width = (text_size.width > i.w + 2) ? text_size.width : (i.w + 2);
			max_width = std::max(max_width, (int)i.w + 2);
			//max_width = std::max(max_width, 283);
			std::string coords_3d;
			if (!std::isnan(i.z_3d)) {
				std::stringstream ss;
				ss << std::fixed << std::setprecision(2) << "x:" << i.x_3d << "m y:" << i.y_3d << "m z:" << i.z_3d << "m ";
				coords_3d = ss.str();
				cv::Size const text_size_3d = getTextSize(ss.str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, 1, 0);
				int const max_width_3d = (text_size_3d.width > i.w + 2) ? text_size_3d.width : (i.w + 2);
				if (max_width_3d > max_width) max_width = max_width_3d;
			}

			cv::rectangle(mat_img, cv::Point2f(std::max((int)i.x - 1, 0), std::max((int)i.y - 35, 0)),
				cv::Point2f(std::min((int)i.x + max_width, mat_img.cols - 1), std::min((int)i.y, mat_img.rows - 1)),
				color, CV_FILLED, 8, 0);
			putText(mat_img, obj_name, cv::Point2f(i.x, i.y - 16), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(0, 0, 0), 2);
			if (!coords_3d.empty()) putText(mat_img, coords_3d, cv::Point2f(i.x, i.y - 1), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(0, 0, 0), 1);
		}
	}
	if (current_det_fps >= 0 && current_cap_fps >= 0) {
		std::string fps_str = "FPS detection: " + std::to_string(current_det_fps) + "   FPS capture: " + std::to_string(current_cap_fps);
		putText(mat_img, fps_str, cv::Point2f(10, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(50, 255, 0), 2);
	}
}
#endif    // OPENCV


void show_console_result(std::vector<bbox_t> const result_vec, std::vector<std::string> const obj_names, int frame_id = -1) {
	if (frame_id >= 0) std::cout << " Frame: " << frame_id << std::endl;
	for (auto &i : result_vec) {
		if (obj_names.size() > i.obj_id) std::cout << obj_names[i.obj_id] << " - ";
		std::cout << "obj_id = " << i.obj_id << ",  x = " << i.x << ", y = " << i.y
			<< ", w = " << i.w << ", h = " << i.h
			<< std::setprecision(3) << ", prob = " << i.prob << std::endl;
	}
}

std::vector<std::string> objects_names_from_file(std::string const filename) {
	std::ifstream file(filename);
	std::vector<std::string> file_lines;
	if (!file.is_open()) return file_lines;
	for (std::string line; getline(file, line);) file_lines.push_back(line);
	std::cout << "object names loaded \n";
	return file_lines;
}

template<typename T>
class send_one_replaceable_object_t {
	const bool sync;
	std::atomic<T *> a_ptr;
public:

	void send(T const& _obj) {
		T *new_ptr = new T;
		*new_ptr = _obj;
		if (sync) {
			while (a_ptr.load()) std::this_thread::sleep_for(std::chrono::milliseconds(3));
		}
		std::unique_ptr<T> old_ptr(a_ptr.exchange(new_ptr));
	}

	T receive() {
		std::unique_ptr<T> ptr;
		do {
			while (!a_ptr.load()) std::this_thread::sleep_for(std::chrono::milliseconds(3));
			ptr.reset(a_ptr.exchange(NULL));
		} while (!ptr);
		T obj = *ptr;
		return obj;
	}

	bool is_object_present() {
		return (a_ptr.load() != NULL);
	}

	send_one_replaceable_object_t(bool _sync) : sync(_sync), a_ptr(NULL)
	{}
};

int main(int argc, char *argv[])
{
	std::string  names_file = "E:/MyFiles/desktop/model_weight/data/kitti.names";
	std::string  cfg_file = "E:/MyFiles/desktop/model_weight/cfg/yolov3-RGB.cfg";
	//std::string  weights_file = "E:/MyFiles/desktop/model_weight/backup/hsv-depth/yolov3-RGB_last.weights";
	std::string  weights_file = "E:/MyFiles/desktop/model_weight/backup/RGB_weight/yolov3-RGB_25200.weights";
	std::string filename = "E:/MyFiles/desktop/testimage/000012.png";
	std::string filename1 = "E:/MyFiles/desktop/testimage/000013.png";

	if (argc > 4) {    //voc.names yolo-voc.cfg yolo-voc.weights test.mp4
		names_file = argv[1];
		cfg_file = argv[2];
		weights_file = argv[3];
		filename = argv[4];
	}
	else if (argc > 1) filename = argv[1];

	float const thresh = (argc > 5) ? std::stof(argv[5]) : 0.2;

	Detector detector(cfg_file, weights_file);
	
	
	
	auto obj_names = objects_names_from_file(names_file);
	std::string out_videofile = "result.avi";
	bool const save_output_videofile = false;   // true - for history
	bool const send_network = false;        // true - for remote detection
	bool const use_kalman_filter = false;   // true - for stationary camera

	bool detection_sync = true;             // true - for video-file
#ifdef TRACK_OPTFLOW    // for slow GPU
	detection_sync = false;
	Tracker_optflow tracker_flow;
	//detector.wait_stream = true;
#endif  // TRACK_OPTFLOW


	cv::Mat mat_img = cv::imread(filename);
	cv::Mat mat_img1 = cv::imread(filename1);

	auto start = std::chrono::steady_clock::now();
	std::vector<bbox_t> result_vec = detector.detect(mat_img);

	/*detector.~Detector();
	Detector detector1("E:/MyFiles/desktop/model_weight/cfg/yolov3-fdepth.cfg", "E:/MyFiles/desktop/model_weight/backup/depth_g/yolov3-fdepth_24200.weights");*/

	std::vector<bbox_t> result_vec1 = detector.detect(mat_img1);
	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> spent = end - start;
	std::cout << " Time: " << spent.count() /2<< " sec \n";


	//string refFilename("E:/MyFiles/desktop/testimage/2/000002_11.png");
	//cout << "Reading reference image : " << refFilename << endl;
	//Mat imReference = imread(refFilename);


	//// Read image to be aligned
	//string imFilename("E:/MyFiles/desktop/testimage/2/000002_10.png");
	//cout << "Reading image to align : " << imFilename << endl;
	//Mat im = imread(imFilename);


	// Registered image will be resotred in imReg. 
	// The estimated homography will be stored in h. 
	Mat imReg, h;

	// Align images
	cout << "Aligning images ..." << endl;
	alignImages(mat_img, mat_img1, imReg, h, result_vec1, result_vec);
	cvtColor(imReg, imReg, CV_BGR2GRAY);
	cvtColor(mat_img1, mat_img1, CV_BGR2GRAY);
	Mat dif;
	absdiff(imReg, mat_img1, dif);


	//result_vec = detector.tracking_id(result_vec);    // comment it - if track_id is not required
	//bbox_t i = result_vec[0];
	//bbox_t i1 = result_vec1[0];

	//int x = (i.x > i1.x) ? i1.x : i.x;
	//int y = (i.y > i1.y) ? i1.y : i.y;
	//int h = (i.h > i1.h) ? i.h : i1.h;
	//h = int(h*1.2);
	//int w = (i.w > i1.w) ? i.w : i1.w;
	//w = int(w*1.2);

	//cv::Mat image_roi= mat_img(cv::Rect(x,y,w,h));
	//cv::Mat image_roi1 = mat_img1(cv::Rect(x, y, w, h));

	//Mat of = cv::optflow::readOpticalFlow("E:/MyFiles/desktop/testimage/84/Brox.flo");
	//Mat planes[2];
	//cv::split(of, planes);
	//Mat flowx(planes[0]);
	//Mat flowy(planes[1]);

	//将光流中的目标分割保存
	//Mat roi_of = optic(mat_img, mat_img1);

	//ROI_cut(roi_of, result_vec);
	//Mat roi_of1 = optic(mat_img, mat_img1);
	//draw_boxes(flowx, result_vec, obj_names);
	//draw_boxes(flowy, result_vec, obj_names);
	//draw_boxes(mat_img, result_vec, obj_names);
	draw_boxes(mat_img1, result_vec1, obj_names);
	//result_confus(result_vec, result_vec1);

	//计算IoU
	//std::vector<bbox_t> end_vec = result_confus(result_vec, result_vec1);
	draw_boxes(mat_img, result_vec, obj_names);
	cv::imshow("Detected", mat_img);
	//cv::imshow("FB_flow", roi_of);
	show_console_result(result_vec, obj_names);
	cv::waitKey(0);

	return 0;
}
