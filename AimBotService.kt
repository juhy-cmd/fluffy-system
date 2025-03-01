package com.example.aimbot

import android.accessibilityservice.AccessibilityService
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.PixelFormat
import android.hardware.display.DisplayManager
import android.media.ImageReader
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.view.Gravity
import android.view.LayoutInflater
import android.view.MotionEvent
import android.view.View
import android.view.WindowManager
import android.widget.Button
import com.example.aimbot.ml.YoloV8Tflite
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.TensorImage
import java.io.ByteBuffer

class AimBotService : AccessibilityService() {
    private lateinit var windowManager: WindowManager
    private lateinit var overlayView: View
    private lateinit var yoloModel: YoloV8Tflite
    private lateinit var imageReader: ImageReader
    private val handlerThread = HandlerThread("ScreenCaptureThread")
    private lateinit var handler: Handler
    
    override fun onServiceConnected() {
        super.onServiceConnected()
        setupOverlay()
        setupScreenCapture()
        loadModel()
    }

    private fun setupOverlay() {
        windowManager = getSystemService(Context.WINDOW_SERVICE) as WindowManager
        overlayView = LayoutInflater.from(this).inflate(R.layout.overlay, null)
        val layoutParams = WindowManager.LayoutParams(
            WindowManager.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.TYPE_ACCESSIBILITY_OVERLAY,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE,
            PixelFormat.TRANSLUCENT
        )
        layoutParams.gravity = Gravity.TOP or Gravity.START
        windowManager.addView(overlayView, layoutParams)
        overlayView.findViewById<Button>(R.id.toggle_button).setOnClickListener {
            startAimbot()
        }
    }

    private fun setupScreenCapture() {
        handlerThread.start()
        handler = Handler(handlerThread.looper)
        imageReader = ImageReader.newInstance(1080, 1920, PixelFormat.RGBA_8888, 2)
        val displayManager = getSystemService(Context.DISPLAY_SERVICE) as DisplayManager
        val display = displayManager.getDisplay(DisplayManager.DEFAULT_DISPLAY)
        display?.let {
            imageReader.setOnImageAvailableListener({ reader ->
                val image = reader.acquireLatestImage() ?: return@setOnImageAvailableListener
                val bitmap = imageToBitmap(image)
                detectAndAim(bitmap)
                image.close()
            }, handler)
        }
    }

    private fun imageToBitmap(image: android.media.Image): Bitmap {
        val buffer: ByteBuffer = image.planes[0].buffer
        val bytes = ByteArray(buffer.remaining())
        buffer.get(bytes)
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
    }

    private fun loadModel() {
        yoloModel = YoloV8Tflite.newInstance(this)
    }

    private fun detectAndAim(bitmap: Bitmap) {
        val tensorImage = TensorImage(DataType.UINT8)
        tensorImage.load(bitmap)
        val outputs = yoloModel.process(tensorImage)
        val detections = outputs.detectionResultList
        if (detections.isNotEmpty()) {
            val target = detections[0]
            Log.d("Aimbot", "Target detected at: ${target.boundingBox.centerX()}, ${target.boundingBox.centerY()}")
            moveAim(target)
        }
    }

    private fun moveAim(target: YoloV8Tflite.DetectionResult) {
        val x = target.boundingBox.centerX()
        val y = target.boundingBox.centerY()
        Log.d("Aimbot", "Moving aim to: $x, $y")
        val action = MotionEvent.obtain(System.currentTimeMillis(), System.currentTimeMillis(),
            MotionEvent.ACTION_DOWN, x.toFloat(), y.toFloat(), 0)
        dispatchTouchEvent(action)
        val actionMove = MotionEvent.obtain(System.currentTimeMillis(), System.currentTimeMillis(),
            MotionEvent.ACTION_MOVE, x.toFloat(), y.toFloat(), 0)
        dispatchTouchEvent(actionMove)
        val actionUp = MotionEvent.obtain(System.currentTimeMillis(), System.currentTimeMillis(),
            MotionEvent.ACTION_UP, x.toFloat(), y.toFloat(), 0)
        dispatchTouchEvent(actionUp)
    }

    private fun startAimbot() {
        Log.d("Aimbot", "Aimbot started")
        setupScreenCapture()
    }
}
