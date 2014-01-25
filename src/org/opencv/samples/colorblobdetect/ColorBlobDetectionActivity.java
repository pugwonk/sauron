package org.opencv.samples.colorblobdetect;

import java.util.ArrayList;
import java.util.List;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.imgproc.Imgproc;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.view.Window;
import android.view.WindowManager;

public class ColorBlobDetectionActivity extends Activity implements
		CvCameraViewListener2 {
	private static final String TAG = "OCVSample::Activity";

	private Mat mRgba;
	private Mat mRgbaPrev;

	private CameraBridgeViewBase mOpenCvCameraView;

	private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
		@Override
		public void onManagerConnected(int status) {
			switch (status) {
			case LoaderCallbackInterface.SUCCESS: {
				Log.i(TAG, "OpenCV loaded successfully");
				mOpenCvCameraView.enableView();
			}
				break;
			default: {
				super.onManagerConnected(status);
			}
				break;
			}
		}
	};

	public ColorBlobDetectionActivity() {
		Log.i(TAG, "Instantiated new " + this.getClass());
	}

	/** Called when the activity is first created. */
	@Override
	public void onCreate(Bundle savedInstanceState) {
		Log.i(TAG, "called onCreate");
		super.onCreate(savedInstanceState);
		requestWindowFeature(Window.FEATURE_NO_TITLE);
		getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

		setContentView(R.layout.color_blob_detection_surface_view);

		mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.color_blob_detection_activity_surface_view);
		mOpenCvCameraView.setMaxFrameSize(200, 200);
		mOpenCvCameraView.setCvCameraViewListener(this);
	}

	@Override
	public void onPause() {
		super.onPause();
		if (mOpenCvCameraView != null)
			mOpenCvCameraView.disableView();
	}

	@Override
	public void onResume() {
		super.onResume();
		OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_3, this,
				mLoaderCallback);
	}

	public void onDestroy() {
		super.onDestroy();
		if (mOpenCvCameraView != null)
			mOpenCvCameraView.disableView();
	}

	public void onCameraViewStarted(int width, int height) {
		mRgba = new Mat(height, width, CvType.CV_8U);
	}

	public void onCameraViewStopped() {
		mRgba.release();
	}

	public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
		// Some significant thievery from
		// https://code.google.com/p/javacv/source/browse/samples/MotionDetector.java?r=02385ce192fb82f1668386e55ff71ed8d6f88ae3

		mRgba = inputFrame.rgba();

		// Make BW version of frame
		Mat bw = new Mat(mRgba.height(), mRgba.width(), CvType.CV_8UC1);
		Imgproc.cvtColor(mRgba, bw, Imgproc.COLOR_RGB2GRAY);
		// The blur was from the code above - not entirely sure whether it helps
		// things much or not but it does seem to stop some camera noise from
		// triggering the detector
		Imgproc.GaussianBlur(bw, bw, new Size(5, 5), 0);

		// Prepare diff frame
		Mat diff = new Mat(mRgba.height(), mRgba.width(), CvType.CV_8UC1);
		if (mRgbaPrev != null) {
			Core.absdiff(bw, mRgbaPrev, diff);
			// Do some noise reduction
			Imgproc.threshold(diff, diff, 64, 255, Imgproc.THRESH_BINARY);
			List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
			Mat mHierarchy = new Mat(0, 0, CvType.CV_8U);
			Imgproc.findContours(diff, contours, mHierarchy,
					Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

			if (contours.size() > 0) {
				MatOfPoint2f allPoints = new MatOfPoint2f();
				// allPoints.push_back(new
				// MatOfPoint2f(contours.get(0).toArray()));
				// Bumble through all the contours
				for (MatOfPoint matOfPoint : contours) {
					MatOfPoint2f points = new MatOfPoint2f(matOfPoint.toArray());
					allPoints.push_back(points);
					// Draw a box around this specific area
					RotatedRect box = Imgproc.minAreaRect(points);
					Core.rectangle(mRgba, box.boundingRect().tl(), box
							.boundingRect().br(), new Scalar(180, 0, 0));
				}
				// Draw a box around the whole lot
				RotatedRect box = Imgproc.minAreaRect(allPoints);
				Core.rectangle(mRgba, box.boundingRect().tl(), box
						.boundingRect().br(), new Scalar(0, 0, 250));
			}

			mRgbaPrev.release();
			diff.release();
		}

		mRgbaPrev = bw;
		return mRgba;
	}
}
