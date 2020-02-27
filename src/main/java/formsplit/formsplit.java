package formsplit;

import java.util.LinkedList;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.HighGui;

import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class formsplit {

	static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);  //º”‘ÿ∂ØÃ¨¡¥Ω”ø‚
    }
	
	public static void FindContours(Mat srcImg, 
			Mat netImg, List<MatOfPoint> contours, Mat hierarchy) {
		Mat grayImg = Mat.zeros(srcImg.size(), CvType.CV_64FC1);
		Mat gaussImg = grayImg;
		Mat notImg = grayImg;
		Mat adaptiveThres = grayImg;
		Mat mask = grayImg;
		
		Imgproc.cvtColor(srcImg, grayImg, Imgproc.COLOR_BGR2GRAY);
		
		Imgproc.GaussianBlur(grayImg, gaussImg, new Size(3, 3), 0);
		
		Core.bitwise_not(gaussImg, notImg);
		
		Imgproc.adaptiveThreshold(notImg, adaptiveThres, 255, 
				Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 5, -2);
	
//		HighGui.imshow("threshold", adaptiveThres);
//		HighGui.waitKey();
		
		Mat horizontal = Mat.zeros(srcImg.size(), CvType.CV_16FC1);
		Mat vertical = Mat.zeros(srcImg.size(), CvType.CV_16FC1);
		adaptiveThres.copyTo(horizontal);
		adaptiveThres.copyTo(vertical);
		int scale = 8;
		
		int horizontalSize = horizontal.cols() / scale;
		Mat horizontalStruct = Imgproc.getStructuringElement(
				Imgproc.MORPH_RECT, new Size(horizontalSize, 1));
		
		Imgproc.erode(horizontal, horizontal, horizontalStruct);
		Imgproc.dilate(horizontal, horizontal, horizontalStruct);
		
		int h_minx, h_maxx, h_miny, h_maxy;
		h_maxx = h_maxy = 0;
		h_minx = h_miny = Math.max(srcImg.rows(), srcImg.cols());
		Mat lines = new Mat();
		Imgproc.HoughLinesP(horizontal, lines, 1, 3.1415926/180, 10, 0, srcImg.cols()*2);
		for(int i = 0; i < lines.rows(); ++i) {
			for(int j = 0; j < lines.cols(); ++j) {
				int[] arr = new int[4];
				lines.get(i, j, arr);

				h_minx = Math.min(h_minx, Math.min(arr[1], arr[3]));
				h_maxx = Math.max(h_maxx, Math.max(arr[1], arr[3]));
				h_miny = Math.min(h_miny, Math.min(arr[0], arr[2]));
				h_maxy = Math.max(h_maxy, Math.max(arr[0], arr[2]));
			}
		}		
		
//		Core.bitwise_not(horizontal, horizontal);
//		Imgproc.erode(horizontal, horizontal, horizontalStruct);
//		Core.bitwise_not(horizontal, horizontal);
		
		int verticalSize = horizontal.cols() / scale;
		Mat verticalStruct = Imgproc.getStructuringElement(
				Imgproc.MORPH_RECT, new Size(1, verticalSize));
		
		Imgproc.erode(vertical, vertical, verticalStruct);
		Imgproc.dilate(vertical, vertical, verticalStruct);
		
		int v_minx, v_maxx, v_miny, v_maxy;
		v_maxx = v_maxy = 0;
		v_minx = v_miny = Math.max(srcImg.cols(), srcImg.cols());
		lines = new Mat();
		Imgproc.HoughLinesP(horizontal, lines, 1, 3.1415926/180, 10, 0, srcImg.cols()*2);
		for(int i = 0; i < lines.rows(); ++i) {
			for(int j = 0; j < lines.cols(); ++j) {
				int[] arr = new int[4];
				lines.get(i, j, arr);

				v_minx = Math.min(v_minx, Math.min(arr[1], arr[3]));
				v_maxx = Math.max(v_maxx, Math.max(arr[1], arr[3]));
				v_miny = Math.min(v_miny, Math.min(arr[0], arr[2]));
				v_maxy = Math.max(v_maxy, Math.max(arr[0], arr[2]));
			}
		}

		
//		Core.bitwise_not(vertical, vertical);
//		Imgproc.erode(vertical, vertical, verticalStruct);
//		Core.bitwise_not(vertical, vertical);
		
//		HighGui.imshow("vertical", vertical);
//		HighGui.waitKey(0);
		
		Core.add(horizontal, vertical, mask);
		
//		System.out.println(h_minx + " " + h_maxx + " " + h_miny + " " + h_maxy);
//		System.out.println(v_minx + " " + v_maxx + " " + v_miny + " " + v_maxy);
//		if(Math.abs(h_minx - v_minx) > 3) {
			Imgproc.line(mask, new Point(h_miny, v_minx), new Point(h_maxy, v_minx), new Scalar(255), 10, Imgproc.LINE_AA);
//		}
//		if(Math.abs(h_maxx - v_maxx) > 3) {
			Imgproc.line(mask, new Point(h_miny, v_maxx), new Point(h_maxy, v_maxx), new Scalar(255), 10, Imgproc.LINE_AA);
//		}
//		if(Math.abs(h_miny - v_miny) > 3) {
			Imgproc.line(mask, new Point(h_miny, v_minx), new Point(h_miny, v_maxx), new Scalar(255), 10, Imgproc.LINE_AA);
//		}
//		if(Math.abs(h_maxy - v_maxy) > 3) {
			Imgproc.line(mask, new Point(h_maxy, v_minx), new Point(h_maxy, v_maxx), new Scalar(255), 10, Imgproc.LINE_AA);
//		}
		
		Imgcodecs.imwrite("D:\\eclipse-workspace\\formsplit\\image\\mask.png", mask);
//		HighGui.imshow("mask", mask);
//		HighGui.waitKey(0);
		
//		Core.bitwise_and(horizontal, vertical, netImg);
//		HighGui.imshow("netImg", netImg);
//		HighGui.waitKey(0);
		HighGui.destroyAllWindows();
		
		Imgproc.findContours(mask, contours, hierarchy, 
				Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
	}
	
	public static void GetSplitImg(Mat srcImg, Mat netImg, 
			List<MatOfPoint> contours, String cutImgName, 
			String resPath, Mat hierarchy) {
		MatOfPoint2f mat2f = new MatOfPoint2f();
		MatOfPoint2f approx = new MatOfPoint2f();
		System.out.println(hierarchy.size());
		for(int i = 0; i < contours.size(); ++i) {
			int[] arr = new int[4];
			hierarchy.get(1, i, arr);
			System.out.println(arr[0] + " " + arr[1] + " " + arr[2] + " " + arr[3]);
			double area = Imgproc.contourArea(contours.get(i));
			Mat recImg = srcImg.clone();
			
			contours.get(i).convertTo(mat2f, CvType.CV_32FC2);
			double epsilon = 0.1 * Imgproc.arcLength(mat2f, true);
			Imgproc.approxPolyDP(mat2f, approx, epsilon, true);
			Rect rec = Imgproc.boundingRect(approx);
			
			Imgproc.rectangle(recImg, rec, new Scalar(0, 255, 0));
			Imgcodecs.imwrite(resPath + "t_" + i + ".png", recImg);
			Mat cutImg = srcImg.submat(rec);
			Imgcodecs.imwrite(resPath + "_" + i + ".png", cutImg);
			System.out.println(resPath + "_" + i + ".png");
//			HighGui.imshow("cutImg", cutImg);
//			HighGui.waitKey();
		}
		HighGui.destroyAllWindows();
	}
	
	
	public static void DoSplit(String imgPath, String resPath) {
		Mat srcImg = Imgcodecs.imread(imgPath);
		srcImg = srcImg.submat(10, srcImg.rows() - 10, 10, srcImg.cols() - 10);
		Mat netImg = new Mat();
		List<MatOfPoint> contours = new LinkedList<MatOfPoint>();
		Mat hierarchy = new Mat();
		
		FindContours(srcImg, netImg, contours, hierarchy);
		
		GetSplitImg(srcImg, netImg, contours, resPath, resPath, hierarchy);
	}
	
	public static void main(String[] args) {
		DoSplit("D:\\eclipse-workspace\\formsplit\\image\\p1.jpg", 
				"D:\\eclipse-workspace\\formsplit\\image\\result\\res");
	}
}
