#include <opencv2/opencv.hpp>
#include <vector>

int main()
{
	cv::VideoCapture cap(0);
	cv::CascadeClassifier faceID;
	cv::CascadeClassifier faceDet;
	cv::namedWindow("WebCam", cv::WINDOW_FREERATIO);
	static const int img_scale = 3.0;
	faceID.load("models/haarcascade_eye.xml");
	//faceDet.load("models/haarcascade_frontalface_alt.xml");
	faceDet.load("models/Rubix_cube_classifier.xml");

	for (;;) {
		cv::Mat frame, frame2;
		cap >> frame;
		cv::extractChannel(frame, frame2, 1);
		cv::resize(frame2, frame2, cv::Size(frame2.size().width / img_scale, frame2.size().height / img_scale));

		std::vector<cv::Rect> eyes, faces;
		faceID.detectMultiScale(frame2, eyes, 1.1, 3, 0, cv::Size(20, 20));
		faceDet.detectMultiScale(frame2, faces, 1.1, 3, 0, cv::Size(20, 20));

		for (cv::Rect rc : eyes) {
			cv::Scalar drawColor(0, 0, 255);
			cv::rectangle(frame, cv::Point(cvRound(rc.x * img_scale), cvRound(rc.y * img_scale)),
				cv::Point(cvRound(rc.x + rc.width - 1)*img_scale, cvRound(rc.y + rc.height - 1)*img_scale), drawColor,2,2,0);
			cv::putText(frame, "Eye", cv::Point((rc.x + rc.width) * img_scale, (rc.y + rc.height)*img_scale), 1, 1, drawColor);
		}
		for (cv::Rect rc : faces) {
			cv::Scalar drawColor(0, 0, 255);
			cv::rectangle(frame, cv::Point(cvRound(rc.x * img_scale), cvRound(rc.y * img_scale)),
				cv::Point(cvRound(rc.x + rc.width - 1) * img_scale, cvRound(rc.y + rc.height - 1) * img_scale), drawColor,2,8,0);
			cv::putText(frame, "Rubik's Cube", cv::Point((rc.x + rc.width) * img_scale, (rc.y + rc.height) * img_scale), 1, 1, drawColor);
		}

		cv::imshow("WebCam", frame);
		if (cv::waitKey(30) >= 0) break;
	}
}