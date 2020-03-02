package formsplit;

import java.util.ArrayList;
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

import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

class Position {
	private int x;
	private int y;
	
	public Position(int x, int y) {
		this.setX(x);
		this.setY(y);
	}

	public Position() {
		setX(0);
		setY(0);
	}

	public int getX() {
		return x;
	}

	public void setX(int x) {
		this.x = x;
	}

	public int getY() {
		return y;
	}

	public void setY(int y) {
		this.y = y;
	}
}

class RectangleArea {
	private Position startPos = new Position();
	private int height;
	private int width;
	public  List<String> contents;
	
	public void setPosition(int x, int y) {
		startPos.setX(x);
		startPos.setY(y);
	}
	
	public Position getPosition() {
		return startPos;
	}

	public int getHeight() {
		return height;
	}

	public void setHeight(int height) {
		this.height = height;
	}

	public int getWidth() {
		return width;
	}

	public void setWidth(int width) {
		this.width = width;
	}

}

public class formsplit {
	private static int bound = 25;

	static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);  //º”‘ÿ∂ØÃ¨¡¥Ω”ø‚
    }
	
	public static boolean checkCross(int arr[], Mat lines) {
		int x1, y1, x2, y2;
		x1 = arr[1]; x2 = arr[3];
		int maxx = Math.max(x1, x2);
		int minx = Math.min(x1, x2);
		
		y1 = arr[0]; y2 = arr[2];
		
		for(int i = 0; i < lines.rows(); ++i) {
			int[] pos = new int[4];
			lines.get(i, 0, pos);
			
			if((pos[1] >= minx && pos[1] <= maxx || pos[3] >= minx && pos[1] <= maxx) &&
					(y1 >= pos[0] && y1 <= pos[2] || y2 >= pos[0] && y2 <= pos[2])) {				
				return true;
			}
		}
		return false;
	}
	
	public static int GetLowLine(int x, int y, Mat lines) {
		
		int linex = 10000000;
		
		for(int i = 0; i < lines.rows(); ++i) {
			int[] pos = new int[4];
			lines.get(i, 0, pos);
			if(pos[1] < x || pos[3] < x) continue;
			if(!(Math.min(pos[0], pos[2]) <= y && y <= Math.max(pos[0], pos[2]))) continue;
			int minx = Math.min(pos[1], pos[3]);
			if(minx < linex)linex = minx;
		}
		
		return linex;
	}
	
	public static void findContours(Mat srcImg, 
			Mat netImg, List<MatOfPoint> contours, Mat hierarchy) {
		Mat grayImg = Mat.zeros(srcImg.size(), CvType.CV_64FC1);
		Mat gaussImg = Mat.zeros(srcImg.size(), CvType.CV_16FC1);
		Mat notImg = Mat.zeros(srcImg.size(), CvType.CV_16FC1);
		Mat adaptiveThres = Mat.zeros(srcImg.size(), CvType.CV_16FC1);
		Mat mask = Mat.zeros(srcImg.size(), CvType.CV_8UC1);
		
		Imgproc.cvtColor(srcImg, grayImg, Imgproc.COLOR_BGR2GRAY);
		
		Imgproc.GaussianBlur(grayImg, gaussImg, new Size(3, 3), 0);
		
		Core.bitwise_not(gaussImg, notImg);
		
		Imgproc.adaptiveThreshold(notImg, adaptiveThres, 255, 
				Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 33, -2);
		
		Mat horizontal = Mat.zeros(srcImg.size(), CvType.CV_16FC1);
		Mat vertical = Mat.zeros(srcImg.size(), CvType.CV_16FC1);
		adaptiveThres.copyTo(horizontal);
		adaptiveThres.copyTo(vertical);
		int hScale = 10;
		
		int horizontalSize = horizontal.cols() / hScale;
		Mat horizontalStruct = Imgproc.getStructuringElement(
				Imgproc.MORPH_RECT, new Size(horizontalSize, 1));
		
		Imgproc.erode(horizontal, horizontal, horizontalStruct);
		Imgproc.dilate(horizontal, horizontal, horizontalStruct);
		
		int h_minx, h_maxx, h_miny, h_maxy;
		h_maxx = h_maxy = 0;
		h_minx = h_miny = Math.max(srcImg.rows(), srcImg.cols());
		Mat hlines = new Mat();
		Imgproc.HoughLinesP(horizontal, hlines, 1, 3.1415926/180, 10, 0, 50);
		for(int i = 0; i < hlines.rows(); ++i) {
			int[] arr = new int[4];
			hlines.get(i, 0, arr);

			h_minx = Math.min(h_minx, Math.min(arr[1], arr[3]));
			h_maxx = Math.max(h_maxx, Math.max(arr[1], arr[3]));
			h_miny = Math.min(h_miny, Math.min(arr[0], arr[2]));
			h_maxy = Math.max(h_maxy, Math.max(arr[0], arr[2]));
			
			Imgproc.line(mask, new Point(arr[0], arr[1]), new Point(arr[2], arr[3]), new Scalar(255), 6, Imgproc.LINE_AA);
		}	
		
		int vScale = 63;
		int verticalSize = horizontal.cols() / vScale;
		Mat verticalStruct = Imgproc.getStructuringElement(
				Imgproc.MORPH_RECT, new Size(1, verticalSize));
		
		Imgproc.erode(vertical, vertical, verticalStruct);
		Imgproc.dilate(vertical, vertical, verticalStruct);
		
		int v_minx, v_maxx, v_miny, v_maxy;
		v_maxx = v_maxy = 0;
		v_minx = v_miny = Math.max(srcImg.cols(), srcImg.cols());
		Mat vlines = new Mat();
		Imgproc.HoughLinesP(vertical, vlines, 1, 3.1415926/180, 10, 0, 50);
		
		for(int i = 0; i < vlines.rows(); ++i) {
			int[] arr = new int[4];
			vlines.get(i, 0, arr);

			if(!checkCross(arr, hlines))continue;
						
			if(Math.abs(arr[1] - arr[3]) < 50) {
				
				int lowline = 0;
				if(arr[1] > arr[3]) {
					lowline = GetLowLine(arr[1], arr[0], hlines);
					if(lowline < 10000000)
						Imgproc.line(mask, new Point(arr[0], lowline), new Point(arr[2], arr[3]), new Scalar(255), 6, Imgproc.LINE_AA);
				} else {
					lowline = GetLowLine(arr[3], arr[2], hlines);
					if(lowline < 10000000)
						Imgproc.line(mask, new Point(arr[0], arr[1]), new Point(arr[2], lowline), new Scalar(255), 6, Imgproc.LINE_AA);
				}
				
			} else {
				Imgproc.line(mask, new Point(arr[0], arr[1]), new Point(arr[2], arr[3]), new Scalar(255), 6, Imgproc.LINE_AA);
			}
			
			v_minx = Math.min(v_minx, Math.min(arr[1], arr[3]));
			v_maxx = Math.max(v_maxx, Math.max(arr[1], arr[3]));
			v_miny = Math.min(v_miny, Math.min(arr[0], arr[2]));
			v_maxy = Math.max(v_maxy, Math.max(arr[0], arr[2]));
		}
		
		Imgproc.line(mask, new Point(h_miny, v_minx), new Point(h_maxy, v_minx), new Scalar(255), 10, Imgproc.LINE_AA);
		Imgproc.line(mask, new Point(h_miny, v_maxx), new Point(h_maxy, v_maxx), new Scalar(255), 10, Imgproc.LINE_AA);
		Imgproc.line(mask, new Point(h_miny, v_minx), new Point(h_miny, v_maxx), new Scalar(255), 10, Imgproc.LINE_AA);
		Imgproc.line(mask, new Point(h_maxy, v_minx), new Point(h_maxy, v_maxx), new Scalar(255), 10, Imgproc.LINE_AA);
		
		Imgcodecs.imwrite("D:\\eclipse-workspace\\formsplit\\image\\mask.png", mask);
		
		Imgproc.findContours(mask, contours, hierarchy, 
				Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
	}
	
	public static List<RectangleArea> getSplitImg(Mat srcImg, Mat netImg, 
			List<MatOfPoint> contours, Mat hierarchy) {
		
		List<RectangleArea> ans = new ArrayList<RectangleArea>();
		MatOfPoint2f mat2f = new MatOfPoint2f();
		MatOfPoint2f approx = new MatOfPoint2f();
		for(int i = 0; i < contours.size(); ++i) {
			contours.get(i).convertTo(mat2f, CvType.CV_32FC2);
			double epsilon = 0.1 * Imgproc.arcLength(mat2f, true);
			Imgproc.approxPolyDP(mat2f, approx, epsilon, true);
			Rect rec = Imgproc.boundingRect(approx);
			if(rec.height < 20 || rec.width < 20)continue;
			
			RectangleArea recItem = new RectangleArea();
			recItem.setPosition(rec.x + bound, rec.y + bound);
			recItem.setHeight(rec.height);
			recItem.setWidth(rec.width);
			ans.add(recItem);
			
		}
		
		RectangleArea top = new RectangleArea();
		top.setPosition(0, 0);
		top.setHeight(ans.get(ans.size() - 1).getPosition().getY());
		top.setWidth(srcImg.cols());
		
		RectangleArea bottom = new RectangleArea();
		bottom.setPosition(0, ans.get(0).getPosition().getY() + ans.get(0).getHeight());
		bottom.setHeight(srcImg.rows() - ans.get(0).getHeight() - ans.get(0).getPosition().getY());
		bottom.setWidth(srcImg.cols());
		
		ans.add(top);
		if(bottom.getHeight() > 0)ans.add(bottom);
		
		return ans;
	}
	
	public static List<RectangleArea> doSplit(String imgPath) {
		Mat srcImg = Imgcodecs.imread(imgPath);
		
		srcImg = srcImg.submat(bound, srcImg.rows() - bound, bound, srcImg.cols() - bound);
		Mat netImg = new Mat();
		List<MatOfPoint> contours = new LinkedList<MatOfPoint>();
		Mat hierarchy = new Mat();
		
		findContours(srcImg, netImg, contours, hierarchy);
		
		return getSplitImg(srcImg, netImg, contours, hierarchy);
	}
	
	public static void main(String[] args) {
		
		List<RectangleArea> arr = doSplit("D:\\eclipse-workspace\\formsplit\\image\\wx.png");
		
		Mat srcImg = Imgcodecs.imread("D:\\eclipse-workspace\\formsplit\\image\\wx.png");
		String resPath = "D:\\eclipse-workspace\\formsplit\\image\\result\\res";
		int cnt = 0;
//		Imgproc.line(srcImg, new Point(0,0), new Point(100, 200), new Scalar(255,0,0), 12, Imgproc.LINE_AA);
		for(int i = 0; i < arr.size(); ++i) {
			Mat recImg = srcImg.clone();
			
			Position pos = arr.get(i).getPosition();
			
			Rect rec = new Rect();
			rec.x = pos.getX();
			rec.y = pos.getY();
			rec.height = arr.get(i).getHeight();
			rec.width = arr.get(i).getWidth();
			
			System.out.println(rec.x + "," + rec.y + " " + rec.height + " " + rec.width);
			
			Imgproc.rectangle(recImg, rec, new Scalar(255, 255, 0), 10);
			cnt++;
			Imgcodecs.imwrite(resPath + "t_" + cnt + ".png", recImg);
			Mat cutImg = srcImg.submat(rec);    
			Imgcodecs.imwrite(resPath + "_" + cnt + ".png", cutImg);
//			System.out.println(resPath + "_" + cnt + ".png");
		}
	}
}
