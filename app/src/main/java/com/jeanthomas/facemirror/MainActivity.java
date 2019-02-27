package com.jeanthomas.facemirror;

import android.os.Bundle;

import java.io.File;
import java.io.InputStream;
import java.io.FileOutputStream;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.Objdetect;
import org.opencv.objdetect.CascadeClassifier;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.view.View.OnTouchListener;
import android.view.View.OnClickListener;
import android.view.SurfaceView;
import android.content.Context;
import android.widget.Button;
import android.widget.SeekBar;

public class MainActivity extends Activity implements OnTouchListener, CvCameraViewListener2 {
    private static final String TAG = "MainActivity";

    private boolean side = false;
    private CascadeClassifier classifier;
    private CameraBridgeViewBase mOpenCvCameraView;
    private File mCascadeFile;

    private Mat mRgba;
    private Mat wbFrame;

    private int mCameraId = 0;

    private BaseLoaderCallback  mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                    mOpenCvCameraView.setOnTouchListener(MainActivity.this);
                    try {
                        InputStream is = getResources().openRawResource(R.raw.haarcascade_frontalface_default);
                        File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
                        mCascadeFile = new File(cascadeDir, "haarcascade_frontalface_default.xml");
                        FileOutputStream os = new FileOutputStream(mCascadeFile);

                        byte[] buffer = new byte[4096];
                        int bytesRead;

                        while ((bytesRead = is.read(buffer)) != -1) {
                            os.write(buffer, 0, bytesRead);
                            Log.d(TAG, "buffer: " + buffer.toString());
                        }
                        is.close();
                        os.close();

                        Log.i(TAG, "Cascade temp file path:" + mCascadeFile.getAbsolutePath());
                        classifier = new CascadeClassifier(mCascadeFile.getAbsolutePath());
                        if (classifier.empty()) {
                            Log.e(TAG, "Failed to load cascade classifier");
                            classifier = null;
                        }
                    }
                    catch (Exception e) {
                        Log.e(TAG, "Error loading cascade", e);
                    }
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    public MainActivity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.activity_main);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.color_blob_detection_activity_surface_view);
        mOpenCvCameraView.setMaxFrameSize(800, 600);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);

        Button swapCamBtn = (Button)findViewById(R.id.swapCamButton);
        swapCamBtn.setOnClickListener(new OnClickListener() {
            @Override
            public void onClick(View arg0) {
                mCameraId = mCameraId^1;
                mOpenCvCameraView.disableView();
                mOpenCvCameraView.setCameraIndex(mCameraId);
                mOpenCvCameraView.enableView();
            }
        });

        Button swapMirrorBtn = (Button)findViewById(R.id.swapMirrorButton);
        swapMirrorBtn.setOnClickListener(new OnClickListener() {
            @Override
            public void onClick(View arg0) {
                side = !side;
            }
        });
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {

    }

    public void onCameraViewStopped() {

    }

    public boolean onTouch(View v, MotionEvent event) {
        //side = !side;

        return true;
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        wbFrame = inputFrame.gray();
        //Imgproc.cvtColor(mRgba, wbFrame, Imgproc.COLOR_BGR2GRAY);

        MatOfRect faces = new MatOfRect();

        SeekBar downscaleFactorCtrl = findViewById(R.id.downscaleFactor);
        double scaleFactor = downscaleFactorCtrl.getScaleX() / 100.0f + 1.1f;
        classifier.detectMultiScale(wbFrame, faces, scaleFactor, 3, Objdetect.CASCADE_SCALE_IMAGE, new Size(100, 100));

        for (Rect face : faces.toArray()) {
            if (side) {
                Rect rectLeft = face;
                rectLeft.width /= 2;
                Rect rectRight = rectLeft.clone();
                rectRight.x += rectRight.width;
                Core.flip(mRgba.submat(rectLeft), mRgba.submat(rectRight), 1);
            } else {
                Rect rectRight = face;
                rectRight.width /= 2;
                rectRight.x += rectRight.width;
                Rect rectLeft = rectRight.clone();
                rectLeft.x -= rectLeft.width;
                Core.flip(mRgba.submat(rectRight), mRgba.submat(rectLeft), 1);
            }
        }

        return mRgba;
    }
}
