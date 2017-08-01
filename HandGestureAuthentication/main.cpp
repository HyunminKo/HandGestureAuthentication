#include "opencv2/opencv.hpp"
#include "VideoFaceDetector.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string.h>

using namespace cv;

Mat getHandMask(const Mat& image, int minCr = 128, int maxCr = 170, int minCb = 73, int maxCb = 158);
Point getHandCenter(const Mat& mask, double& radius);
int getFingerCount(const Mat& mask, Point center, double radius, double scale = 1.9);

std::vector<Mat>				planes;
std::vector<std::vector<Point>> contours;
const cv::String				CASCADE_FILE("haarcascade_frontalface_default.xml");

int main(int argc, char** argv) {


	VideoCapture cap;
	VideoFaceDetector detector(CASCADE_FILE, cap);
	if (!cap.open(0))
		return 0;

	double fps = 0, time_per_frame;

	for (;;) {

		Mat frame;
		auto start = cv::getCPUTickCount();
		detector >> frame;

		const string& text = "count: ";
		const string& fpstext = ", fps: ";

		if (detector.isFaceFound())
		{
			cv::rectangle(frame, detector.face(), cv::Scalar(0, 0, 0), CV_FILLED);
			//cv::circle(frame, detector.facePosition(), 30, cv::Scalar(0, 255, 0));
		}

		Mat mask = getHandMask(frame);
		Mat handImage(frame.size(), CV_8UC3, Scalar(0));

		erode(mask, mask, Mat(3, 3, CV_8U, Scalar(1)), Point(-1, -1), 2);
		erode(mask, mask, Mat(3, 3, CV_8U, Scalar(1)), Point(-1, -1), 2);

		add(frame, Scalar(0), handImage, mask);

		double radius;

		Point center = getHandCenter(mask, radius);
		int fingerCount = getFingerCount(mask, center, radius);

		auto end = cv::getCPUTickCount();
		time_per_frame = (end - start) / cv::getTickFrequency();
		fps = (15 * fps + (1 / time_per_frame)) / 16;


		std::cout << "손바닥 중심점 좌표:" << center << ", 반지름:" << radius << ", 손가락 수:" << fingerCount << std::endl;

		//손바닥 중심점 그리기
		circle(frame, center, 2, Scalar(0, 255, 0), -1);

		//손바닥 영역 그리기
		circle(frame, center, (int)(radius * 1.9), Scalar(255, 0, 0), 2);


		putText(frame, text + std::to_string(fingerCount) + fpstext + std::to_string(time_per_frame), Point(10, 30), CV_FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 0, 255), 1);

		imshow("mask", mask);
		imshow("Hand Image", handImage);
		imshow("frame", frame);

		if (waitKey(30) >= 0) break;
	}

	return 0;
}

Mat getHandMask(const Mat& image, int minCr, int maxCr, int minCb, int maxCb) {
	//컬러 공간 변환 BGR->YCrCb
	Mat YCrCb;
	cvtColor(image, YCrCb, CV_BGR2YCrCb);

	//각 채널별로 분리
	split(YCrCb, planes);

	//각 채널별로 화소마다 비교
	Mat mask(image.size(), CV_8U, Scalar(0));   //결과 마스크를 저장할 영상
	uchar* data = (uchar*) mask.data;
	int nr = image.rows;    //전체 행의 수
	int nc = image.cols;

	for (int i = 0; i<nr; i++) {
		uchar* CrPlane = planes[1].ptr<uchar>(i);   //Cr채널의 i번째 행 주소
		uchar* CbPlane = planes[2].ptr<uchar>(i);   //Cb채널의 i번째 행 주소
		for (int j = 0; j<nc; j++) {
			if ((minCr < CrPlane[j]) && (CrPlane[j] <maxCr) && (minCb < CbPlane[j]) && (CbPlane[j] < maxCb))
				data[i * nc + j] = 255;
		}
	}

	return mask;
}

//손바닥의 중심점과 반지름 반환
//입력은 8bit 단일 채널(CV_8U), 반지름을 저장할 double형 변수
Point getHandCenter(const Mat& mask, double& radius) {
	//거리 변환 행렬을 저장할 변수
	Mat dst;
	distanceTransform(mask, dst, CV_DIST_L2, 5);  //결과는 CV_32SC1 타입

												  //거리 변환 행렬에서 값(거리)이 가장 큰 픽셀의 좌표와, 값을 얻어온다.
	int maxIdx[2];    //좌표 값을 얻어올 배열(행, 열 순으로 저장됨)
	minMaxIdx(dst, NULL, &radius, NULL, maxIdx, mask);   //최소값은 사용 X

	return Point(maxIdx[1], maxIdx[0]);
}

//손목을 제거하지 않은 상태에서 손가락 개수 세기
int getFingerCount(const Mat& mask, Point center, double radius, double scale) {
	//손가락 개수를 세기 위한 원 그리기
	Mat cImg(mask.size(), CV_8U, Scalar(0));
	circle(cImg, center, radius*scale, Scalar(255));

	//원의 외곽선을 저장할 벡터

	findContours(cImg, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	if (contours.size() == 0)   //외곽선이 없을 때 == 손 검출 X
		return -1;

	//외곽선을 따라 돌며 mask의 값이 0에서 1로 바뀌는 지점 확인
	int fingerCount = 0;
	for (int i = 1; i<contours[0].size(); i++) {
		Point p1 = contours[0][i - 1];
		Point p2 = contours[0][i];
		if (mask.at<uchar>(p1.y, p1.x) == 0 && mask.at<uchar>(p2.y, p2.x)>1)
			fingerCount++;
	}
	if (fingerCount > 6)
		return -1;

	//손목과 만나는 개수 1개 제외
	return fingerCount - 1;
}
