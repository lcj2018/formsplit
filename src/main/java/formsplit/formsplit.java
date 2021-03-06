package formsplit;

import java.awt.color.ColorSpace;
import java.awt.image.BufferedImage;
import java.awt.image.ColorConvertOp;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import javax.imageio.ImageIO;

import org.opencv.core.*;
import org.opencv.highgui.HighGui;
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
    private Position midPos = new Position();
    private int height;
    private int width;
//    public List<Sentence> contents = new ArrayList<>();
    //private BufferedImage image;
    private String Title;

    public String getTitle() {
        return Title;
    }

    public void setTitle(String title) {
        Title = title;
    }

    public RectangleArea() {
    }

    public RectangleArea(Position startPos, int height, int width) {
        this.startPos = startPos;
        this.height = height;
        this.width = width;
    }

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

    public boolean isInArea(Position pos) {
        if (pos.getX() >= startPos.getX() && pos.getY() >= startPos.getY()) {
            if ((startPos.getX() + width) >= pos.getX() && (startPos.getY() + height) >= pos.getY()) {
                return true;
            } else {
                return false;
            }
        } else {
            return false;
        }
    }

	public Position getMidPos() {
		return midPos;
	}

	public void setMidPos(int x, int y) {
		midPos.setX(x);
		midPos.setY(y);
	}
}

public class formsplit {
    private static int bound = 25;

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static boolean checkCross(int arr[], Mat lines) {	//check a vertical line and horizontal lines
        int x1, y1, x2, y2;
        x1 = arr[1]; x2 = arr[3];
        int maxx = Math.max(x1, x2);
        int minx = Math.min(x1, x2);

        y1 = arr[0]; y2 = arr[2];

        for(int i = 0; i < lines.rows(); ++i) {
            int[] pos = new int[4];
            lines.get(i, 0, pos);

            if((pos[1] >= minx && pos[1] <= maxx || pos[3] >= minx && pos[3] <= maxx) &&
                    (y1 >= Math.min(pos[0], pos[2]) && y1 <= Math.max(pos[0], pos[2]) || y2 >= Math.min(pos[0], pos[2]) && y2 <= Math.max(pos[0], pos[2]))) {
                return true;
            }
        }
        return false;
    }
    
    public static boolean checkCross2(int arr[], Mat lines) {	//check vertical lines and a horizontal line
        int x1, y1, x2, y2;
        y1 = arr[0]; y2 = arr[2];
        int maxy = Math.max(y1, y2);
        int miny = Math.min(y1, y2);

        x1 = arr[1]; x2 = arr[3];

        for(int i = 0; i < lines.rows(); ++i) {
            int[] pos = new int[4];
            lines.get(i, 0, pos);

            if((pos[0] >= miny - 10 && pos[0] <= maxy + 10 || pos[2] >= miny - 10 && pos[2] <= maxy + 10) &&
                    (x1 >= Math.min(pos[1], pos[3]) && x1 <= Math.max(pos[1], pos[3]) || x2 >= Math.min(pos[1], pos[3]) && x2 <= Math.max(pos[1], pos[3]))) {
                return true;
            }
        }
        return false;
    }

	public static boolean checkQualification(int arr[], Mat lines) {
		int x, y;
		if(arr[1] < arr[3]) {
			x = arr[1];
			y = arr[0];
		} else {
			x = arr[3];
			y = arr[2];
		}
		
		for(int i = 0; i < lines.rows(); ++i) {
			int[] pos = new int[4];
			lines.get(i, 0, pos);
			
			if(Math.min(Math.abs(x - pos[1]), Math.abs(x - pos[3])) < 3
					&& Math.min(arr[0], arr[2]) <= y 
					&& y <= Math.max(arr[0], arr[2])) {
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
    
	public static int findDeepPixel(Mat mat, int posX, int posY) {
		
		int maxV = 0;
		for(int x = Math.max(0, posX - 5); x < Math.min(mat.rows(), posX + 5); ++x) {
			for(int y = Math.max(0, posY - 5); y < Math.min(mat.cols(), posY + 5); ++y) {
				byte[] arr = new byte[mat.channels()];
				mat.get(x, y ,arr);
				maxV = Math.max(maxV, arr[0]);
			}
		}
		
//		if(maxV < 113) {
//			System.out.println(maxV);
//			Mat tt = mat.submat(Math.max(0, posX - 5), Math.min(mat.rows()- 1, posX + 5), Math.max(0, posY - 5), Math.min(mat.cols() - 1, posY + 5));
//			HighGui.imshow("mat", tt);
//			HighGui.waitKey();
//		}
		
		return maxV;
	}
	
	public static int getGrayThres(Mat mat, Mat lines) {
		int thres = 0;
		
		int[] grayCnt = new int[257];
		for(int i = 0;i < lines.rows(); ++i) {
			int[] arr = new int[4];
			lines.get(i, 0, arr);
			
			int gray = 0;
			gray = findDeepPixel(mat, arr[1], arr[0]);
			++grayCnt[gray + 128];
			gray = findDeepPixel(mat, arr[3], arr[2]);
			++grayCnt[gray + 128];
			gray = findDeepPixel(mat, (arr[1] + arr[3]) / 2, (arr[0] + arr[2]) / 2);
			++grayCnt[gray + 128];
		}
		
		for(int i = 255; i >= 0; --i) {
			grayCnt[i] += grayCnt[i + 1];
			if(grayCnt[i] > lines.rows() * 2.4) {
				thres = i;
				break;
			}
		}
		
		return thres - 128;
	}
	
	public static double distance(int[] arr) {
		double ans = Math.sqrt((arr[0] - arr[2]) * (arr[0] - arr[2]) 
				+ (arr[1] - arr[3]) * (arr[1] - arr[3]));
		return ans;
	}
	
	public static void horizontalLineMerge(Mat lines) {
		boolean flag;
		int tmp = 0;
		do
		{
			flag = false;
			
			for(int i=0;i<lines.rows();++i) {
				int[] ipos = new int[4];
				lines.get(i, 0, ipos);
				if(ipos[0]>ipos[2]) {
					tmp = ipos[0];
					ipos[2] = ipos[0];
					ipos[0] = tmp;
					
					tmp = ipos[1];
					ipos[1] = ipos[3];
					ipos[3] = tmp;
				}
				for(int j=0;j<lines.rows();++j) {
					int[] jpos = new int[4];
					lines.get(j, 0, jpos);
					if(jpos[0]>jpos[2]) {
						tmp = jpos[0];
						jpos[2] = jpos[0];
						jpos[0] = tmp;
						
						tmp = jpos[1];
						jpos[1] = jpos[3];
						jpos[3] = tmp;
					}
					
					if(!(Math.min(jpos[1], jpos[3])-Math.max(ipos[1], ipos[3]) < 5 &&
							Math.min(jpos[1], jpos[3])-Math.max(ipos[1], ipos[3]) > 0)) continue;
					
					if(jpos[0] < ipos[2] && ipos[2] < jpos[2]) {
						ipos[2] = jpos[2];
						jpos[0] = jpos[1] = jpos[2] = jpos[3] = 0;
						flag = true;
					}
				}
				
				lines.put(i, 0, ipos);
			}
			
		}while(flag);
	}

    public static void findContours(Mat srcImg, Mat mask,
                                    List<MatOfPoint> contours, Mat hierarchy) {
        Mat grayImg = Mat.zeros(srcImg.size(), CvType.CV_64FC1);
        Mat gaussImg = Mat.zeros(srcImg.size(), CvType.CV_16FC1);
        Mat notImg = Mat.zeros(srcImg.size(), CvType.CV_16FC1);
        Mat adaptiveThres = Mat.zeros(srcImg.size(), CvType.CV_16FC1);

        Imgproc.cvtColor(srcImg, grayImg, Imgproc.COLOR_BGR2GRAY);

        Imgproc.GaussianBlur(grayImg, gaussImg, new Size(3, 3), 0);
        Imgcodecs.imwrite("D:\\git_proj\\formsplit\\image\\gaussImg.png", gaussImg);

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
//        int orih_minx, orih_maxx, orih_miny, orih_maxy;
        h_maxx = h_maxy = 0;
        h_minx = h_miny = Math.max(srcImg.rows(), srcImg.cols());
        Mat hlines = new Mat();
        Imgproc.HoughLinesP(horizontal, hlines, 1, 3.1415926/180, 10, 0, 50);
        int threshold = getGrayThres(grayImg, hlines);
        
        horizontalLineMerge(hlines);
        
        int vScale = 63;
//        int vScale = 10;
        int verticalSize = horizontal.cols() / vScale;
        Mat verticalStruct = Imgproc.getStructuringElement(
                Imgproc.MORPH_RECT, new Size(1, verticalSize));

        Imgproc.erode(vertical, vertical, verticalStruct);
        Imgproc.dilate(vertical, vertical, verticalStruct);

        int v_minx, v_maxx, v_miny, v_maxy;
//        int oriv_minx, oriv_maxx, oriv_miny, oriv_maxy;
        v_maxx = v_maxy = 0;
        v_minx = v_miny = Math.max(srcImg.cols(), srcImg.cols());
        Mat vlines = new Mat();
        Imgproc.HoughLinesP(vertical, vlines, 1, 3.1415926/180, 10, 0, 50);
        
        for(int i = 0; i < hlines.rows(); ++i) {
            int[] arr = new int[4];
            hlines.get(i, 0, arr);
            
            if(!checkCross2(arr, vlines))continue;

            h_minx = Math.min(h_minx, Math.min(arr[1], arr[3]));
            h_maxx = Math.max(h_maxx, Math.max(arr[1], arr[3]));
            h_miny = Math.min(h_miny, Math.min(arr[0], arr[2]));
            h_maxy = Math.max(h_maxy, Math.max(arr[0], arr[2]));
            
			int headGray = findDeepPixel(grayImg, arr[1], arr[0]);
			int midGray = findDeepPixel(grayImg, (arr[1] + arr[3])/2, (arr[0] + arr[2])/2);
			int tailGray = findDeepPixel(grayImg, arr[3], arr[2]);
			if(headGray < threshold - 20 || midGray < threshold - 20 || tailGray < threshold - 20) {
				continue;
			}

            Imgproc.line(mask, new Point(arr[0], arr[1]), new Point(arr[2], arr[3]), new Scalar(255), 3, Imgproc.LINE_AA);
        }
        
//        orih_minx = h_minx;orih_maxx = h_maxx;orih_miny = h_miny;orih_maxy = h_maxy;

        for(int i = 0; i < vlines.rows(); ++i) {
            int[] arr = new int[4];
            vlines.get(i, 0, arr);

            if(Math.abs(arr[0] - arr[2]) > 3) continue;
            
            if(!checkCross(arr, hlines))continue;
            
			int headGray = findDeepPixel(grayImg, arr[1], arr[0]);
			int midGray = findDeepPixel(grayImg, (arr[1] + arr[3])/2, (arr[0] + arr[2])/2);
			int tailGray = findDeepPixel(grayImg, arr[3], arr[2]);
			if(headGray < threshold - 20 || midGray < threshold - 20 || tailGray < threshold - 20) {
				continue;
			}

            if(Math.abs(arr[1] - arr[3]) < 50 && distance(arr) > 11 && checkQualification(arr, hlines)) {

                int lowline = 0;
                if(arr[1] > arr[3]) {
                    lowline = GetLowLine(arr[1], arr[0], hlines);
                    if(lowline < 10000000)
                        Imgproc.line(mask, new Point(arr[0], lowline), new Point(arr[2], arr[3]), new Scalar(255), 3, Imgproc.LINE_AA);
                } else {
                    lowline = GetLowLine(arr[3], arr[2], hlines);
                    if(lowline < 10000000)
                        Imgproc.line(mask, new Point(arr[0], arr[1]), new Point(arr[2], lowline), new Scalar(255), 3, Imgproc.LINE_AA);
                }
//                Imgproc.line(mask, new Point(arr[0], arr[1]), new Point(arr[2], arr[3]), new Scalar(255), 3, Imgproc.LINE_AA);
            } else {
                Imgproc.line(mask, new Point(arr[0], arr[1]), new Point(arr[2], arr[3]), new Scalar(255), 3, Imgproc.LINE_AA);
            }

            v_minx = Math.min(v_minx, Math.min(arr[1], arr[3]));
            v_maxx = Math.max(v_maxx, Math.max(arr[1], arr[3]));
            v_miny = Math.min(v_miny, Math.min(arr[0], arr[2]));
            v_maxy = Math.max(v_maxy, Math.max(arr[0], arr[2]));
        }
        
//        oriv_minx = v_minx;oriv_maxx=v_maxx;oriv_miny = v_miny; oriv_maxy = v_maxy;
        
        int range = 6;
        for(int i = 0; i < hlines.rows(); ++i) {
        	int[] arr = new int[4];
        	hlines.get(i, 0, arr);
        	
        	if(arr[1] + range >= h_maxy) {
        		h_maxy = Math.min(h_maxy, arr[1]);
        	}
        	if(arr[3] + range >= h_maxy) {
        		h_maxy = Math.min(h_maxy, arr[3]);
        	}
        	if(arr[1] - range <= h_miny) {
        		h_miny = Math.max(h_miny, arr[1]);
        	}
        	if(arr[3] - range <= h_miny) {
        		h_miny = Math.max(h_miny, arr[3]);
        	}
        }
        
        for(int i = 0; i < vlines.rows(); ++i) {
        	int[] arr = new int[4];
        	hlines.get(i, 0, arr);
        	
        	if(arr[0] + range >= v_maxx) {
        		v_maxx = Math.min(v_maxx, arr[0]);
        	}
        	if(arr[2] + range >= v_maxx) {
        		v_maxx = Math.min(v_maxx, arr[2]);
        	}
//        	if(arr[0] - range <= v_minx) {
//        		v_minx = Math.max(v_minx, arr[0]);
//        	}
//        	if(arr[2] - range <= v_minx) {
//        		v_minx = Math.max(v_minx, arr[2]);
//        	}
        }

        Imgproc.line(mask, new Point(h_miny, v_minx), new Point(h_maxy, v_minx), new Scalar(255), 3, Imgproc.LINE_AA);
        Imgproc.line(mask, new Point(h_miny, v_maxx), new Point(h_maxy, v_maxx), new Scalar(255), 3, Imgproc.LINE_AA);
        Imgproc.line(mask, new Point(h_miny, v_minx), new Point(h_miny, v_maxx), new Scalar(255), 3, Imgproc.LINE_AA);
        Imgproc.line(mask, new Point(h_maxy, v_minx), new Point(h_maxy, v_maxx), new Scalar(255), 3, Imgproc.LINE_AA);

//        Imgcodecs.imwrite("D:\\git_proj\\formsplit\\image\\mask.png", mask);

        Imgproc.findContours(mask, contours, hierarchy,
                Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
    }

    public static List<RectangleArea> getSplitImg(Mat srcImg, Mat mask,
                                                  List<MatOfPoint> contours, Mat hierarchy) {

        List<RectangleArea> ans = new ArrayList<RectangleArea>();
        MatOfPoint2f mat2f = new MatOfPoint2f();
        MatOfPoint2f approx = new MatOfPoint2f();
        for(int i = 0; i < contours.size(); ++i) {

            int[] arr = new int[4];
            hierarchy.get(0, i, arr);
            if(arr[2] != -1) continue;
            //System.out.println(arr[0] + " " + arr[1] + " " + arr[2] + " " + arr[3] + " " + hierarchy.size() + " " + hierarchy.channels());

            contours.get(i).convertTo(mat2f, CvType.CV_32FC2);
            double epsilon = 0.1 * Imgproc.arcLength(mat2f, true);
            Imgproc.approxPolyDP(mat2f, approx, epsilon, true);
            Rect rec = Imgproc.boundingRect(approx);
            if(rec.height < 20 || rec.width < 20)continue;

            RectangleArea recItem = new RectangleArea();
            recItem.setPosition(rec.x + bound, rec.y + bound);
            recItem.setHeight(rec.height);
            recItem.setWidth(rec.width);
            
            //recItem.setMidPos(rec.x + bound + rec.width / 2, rec.y + bound + rec.height / 2);
            Integer midX = rec.x + bound + rec.width / 2;
            Integer midY = rec.y + bound + rec.height / 2;
            String text = midX.toString() + ',' + midY.toString();
            Point point = new Point(rec.x + rec.width/2 - 55, rec.y + bound + rec.height/2 - 15);
//            Imgproc.putText(mask, text, point, Imgproc.FONT_HERSHEY_SIMPLEX, 0.8, new Scalar(255), 2);

            ans.add(recItem);

        }
        
        Imgcodecs.imwrite("D:\\git_proj\\formsplit\\image\\maskpos.png", mask);

        return ans;
    }
    
    private static BufferedImage convertTo3ByteBGRType(BufferedImage image) {
        BufferedImage convertedImage = new BufferedImage(image.getWidth(), image.getHeight(),
                BufferedImage.TYPE_3BYTE_BGR);
        convertedImage.getGraphics().drawImage(image, 0, 0, null);
        return convertedImage;
    }
    
    public static Mat BufImg2Mat(BufferedImage image) {
        image = convertTo3ByteBGRType(image);
        byte[] data = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        Mat mat = new Mat(image.getHeight(), image.getWidth(), CvType.CV_8UC3);
        mat.put(0, 0, data);
        return mat;
    }
    
    public static BufferedImage toGray(BufferedImage srcImg) {      
        return new ColorConvertOp(ColorSpace.getInstance(ColorSpace.CS_GRAY), null).filter(srcImg, null);
    }
    
    public static BufferedImage BGR2Gray(BufferedImage srcImg) {
    	Mat img = BufImg2Mat(srcImg);
    	Imgproc.cvtColor(img, img, Imgproc.COLOR_BGR2GRAY);
    	return (BufferedImage)HighGui.toBufferedImage(img);
    }

    public static List<RectangleArea> doSplit(BufferedImage img) throws IOException {
        Mat srcImg = BufImg2Mat(img);

        srcImg = srcImg.submat(bound, srcImg.rows() - bound, bound, srcImg.cols() - bound);
        List<MatOfPoint> contours = new LinkedList<MatOfPoint>();
        Mat hierarchy = new Mat();
        Mat mask = Mat.zeros(srcImg.size(), CvType.CV_8UC1);

        findContours(srcImg, mask, contours, hierarchy);

        return getSplitImg(srcImg, mask, contours, hierarchy);
    }

    public static void main(String[] args) throws IOException {
    	File file = new File("D:\\git_proj\\formsplit\\image\\bill.png");
//    	File file = new File("D:\\git_proj\\formsplit\\image\\p1.jpg");
    	BufferedImage img = ImageIO.read(file);

        List<RectangleArea> arr = doSplit(img);

        File grayFile = new File("D:\\git_proj\\formsplit\\image\\9530.png");
        ImageIO.write(toGray(img), "png", grayFile);
        
        Mat srcImg = Imgcodecs.imread("D:\\git_proj\\formsplit\\image\\bill.png");
//        Mat srcImg = Imgcodecs.imread("D:\\git_proj\\formsplit\\image\\p1.jpg");
        String resPath = "D:\\git_proj\\formsplit\\image\\result\\res";
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

//            System.out.println(pos.getX() + "," + pos.getY() + " " + rec.width + " " + rec.height);

            Imgproc.rectangle(recImg, rec, new Scalar(255, 255, 0), 10);
            cnt++;
            Imgcodecs.imwrite(resPath + "t_" + cnt + ".png", recImg);
            Mat cutImg = srcImg.submat(rec);
            Imgcodecs.imwrite(resPath + "_" + cnt + ".png", cutImg);
//            System.out.println(resPath + "_" + cnt + ".png");
        }
    }
}
