package com.plmun.docOCR

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.res.AssetFileDescriptor
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.enableEdgeToEdge
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.max
import kotlin.math.min

// Add this class to track performance metrics
class PerformanceTracker {
    private val timings = mutableMapOf<String, Long>()

    fun startTimer(key: String) {
        timings[key] = System.currentTimeMillis()
    }

    fun stopTimer(key: String): Long {
        val startTime = timings[key] ?: return 0
        return System.currentTimeMillis() - startTime
    }

    fun logMetric(tag: String, metric: String, value: Any) {
        Log.d(tag, "📊 $metric: $value")
    }
}

class MainActivity : ComponentActivity() {

    private lateinit var imgPreview: ImageView
    private lateinit var imgDefault: ImageView
    private lateinit var txtUpload: TextView
    private lateinit var txtCapture: TextView
    private lateinit var btnPredict: Button
    private lateinit var txtResult: TextView
    private lateinit var txtAccuracy: TextView
    private lateinit var tflite: Interpreter

    private var currentBitmap: Bitmap? = null

    companion object {
        private const val TAG = "MainActivity"
        private const val IMAGE_WIDTH = 160
        private const val IMAGE_HEIGHT = 64
        private const val TIME_STEPS = 20
        private const val NUM_CLASSES = 69

        // Full character set from Python
        private val CHARACTERS = (
                "abcdefghijklmnopqrstuvwxyz" +
                        "ABCDEFGHIJKLMNOPQRSTUVWXYZ" +
                        "0123456789" +
                        " -'.,:"
                ).substring(0, NUM_CLASSES - 1)  // Use first 68 characters

        private const val BLANK_TOKEN = NUM_CLASSES - 1
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)

        imgPreview = findViewById(R.id.img_preview)
        imgDefault = findViewById(R.id.img_default)
        txtUpload = findViewById(R.id.txt_upload)
        txtCapture = findViewById(R.id.txt_capture)
        btnPredict = findViewById(R.id.btn_predict)
        txtResult = findViewById(R.id.txt_result)
        txtAccuracy = findViewById(R.id.textView5)

        // Initialize with default state
        resetPredictionResults()

        // Load TFLite model
        try {
            tflite = Interpreter(loadModelFile())
            Log.d(TAG, "✅ TensorFlow Lite Interpreter loaded successfully")

            val inputTensor = tflite.getInputTensor(0)
            Log.d(TAG, "📊 Model information:")
            Log.d(TAG, "  Input shape: ${inputTensor.shape().contentToString()}")
            Log.d(TAG, "  Expected output: [1, $TIME_STEPS, $NUM_CLASSES]")

            // Test the model
            testModelWithDummyInput()

        } catch (e: Exception) {
            Log.e(TAG, "❌ Failed to load TensorFlow Lite model", e)
            Toast.makeText(this, "Failed to load OCR model", Toast.LENGTH_LONG).show()
        }

        // Upload button click - only uploads/previews image
        txtUpload.setOnClickListener {
            val intent = Intent(Intent.ACTION_PICK)
            intent.type = "image/*"
            galleryLauncher.launch(intent)
        }

        // Capture button click - only captures/previews image
        txtCapture.setOnClickListener {
            requestCameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }

        // Predict button click - processes the uploaded/captured image
        btnPredict.setOnClickListener {
            currentBitmap?.let {
                runOcr(it)
            } ?: run {
                Toast.makeText(this, "Please upload or capture an image first", Toast.LENGTH_SHORT).show()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        if (::tflite.isInitialized) tflite.close()
    }

    private fun loadModelFile(): MappedByteBuffer {
        val fileDescriptor: AssetFileDescriptor = assets.openFd("ml/ocr_model_production_fp16.tflite")
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun testModelWithDummyInput() {
        try {
            val dummyBuffer = ByteBuffer.allocateDirect(1 * IMAGE_HEIGHT * IMAGE_WIDTH * 1 * 4)
            dummyBuffer.order(ByteOrder.nativeOrder())
            for (i in 0 until IMAGE_HEIGHT * IMAGE_WIDTH) {
                dummyBuffer.putFloat(1.0f)
            }
            dummyBuffer.rewind()

            val output = Array(1) { Array(TIME_STEPS) { FloatArray(NUM_CLASSES) } }
            tflite.run(dummyBuffer, output)

            Log.d(TAG, "✅ Model test successful")
            Log.d(TAG, "Output shape confirmed: 1 x $TIME_STEPS x $NUM_CLASSES")

        } catch (e: Exception) {
            Log.e(TAG, "❌ Model test failed", e)
        }
    }

    private fun resetPredictionResults() {
        runOnUiThread {
            txtResult.text = "No prediction yet"
            txtAccuracy.text = "0%"
            btnPredict.isEnabled = false
        }
    }

    private fun updateImagePreview(bitmap: Bitmap) {
        runOnUiThread {
            imgDefault.visibility = ImageView.GONE
            imgPreview.visibility = ImageView.VISIBLE
            imgPreview.setImageBitmap(bitmap)
            btnPredict.isEnabled = true

            // Reset prediction results when new image is loaded
            txtResult.text = "Ready for prediction"
            txtAccuracy.text = "0%"
        }
    }

    private val requestCameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) openCamera()
        else Toast.makeText(this, "Camera permission denied", Toast.LENGTH_SHORT).show()
    }

    private fun openCamera() {
        val cameraIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        cameraLauncher.launch(cameraIntent)
    }

    private val galleryLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == Activity.RESULT_OK) {
            val uri = result.data?.data
            try {
                val stream = contentResolver.openInputStream(uri!!)
                val bitmap = BitmapFactory.decodeStream(stream)
                stream?.close()
                currentBitmap = bitmap
                updateImagePreview(bitmap)
            } catch (e: Exception) {
                Log.e(TAG, "Error loading image from gallery", e)
                Toast.makeText(this, "Error loading image", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private val cameraLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == Activity.RESULT_OK) {
            val bmp = result.data?.extras?.get("data") as? Bitmap
            if (bmp != null) {
                currentBitmap = bmp
                updateImagePreview(bmp)
            } else {
                Toast.makeText(this, "Error capturing image", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun runOcr(bitmap: Bitmap) {
        val tracker = PerformanceTracker()

        try {
            tracker.startTimer("total_ocr")
            Log.d(TAG, "STEP 1: Starting OCR processing")

            // Track memory before processing
            val runtime = Runtime.getRuntime()
            val memoryBefore = runtime.totalMemory() - runtime.freeMemory()
            tracker.logMetric(TAG, "Memory before", "${memoryBefore / 1024} KB")

            // Show processing message
            runOnUiThread {
                txtResult.text = "Processing..."
                txtAccuracy.text = "0%"
                btnPredict.isEnabled = false
            }

            // Preprocess image with timing
            tracker.startTimer("preprocessing")
            val grayBitmap = convertToGrayscale(bitmap)
            val resized = Bitmap.createScaledBitmap(grayBitmap, IMAGE_WIDTH, IMAGE_HEIGHT, true)
            val inputBuffer = convertBitmapToByteBuffer(resized)
            val preprocessTime = tracker.stopTimer("preprocessing")
            tracker.logMetric(TAG, "Preprocess time", "${preprocessTime}ms")

            // Create output buffer
            val output = Array(1) { Array(TIME_STEPS) { FloatArray(NUM_CLASSES) } }

            // Run inference with timing
            tracker.startTimer("inference")
            tflite.run(inputBuffer, output)
            val inferenceTime = tracker.stopTimer("inference")
            tracker.logMetric(TAG, "Inference time", "${inferenceTime}ms")
            Log.d(TAG, "STEP 2: Inference completed")

            // Decode CTC output
            tracker.startTimer("decoding")
            val decodedText = decodeCtcOutput(output[0])
            val decodeTime = tracker.stopTimer("decoding")
            tracker.logMetric(TAG, "Decode time", "${decodeTime}ms")
            Log.d(TAG, "STEP 3: Decoded text: '$decodedText' (${decodedText.length} chars)")

            // Calculate confidence
            val confidence = calculateConfidence(output[0])

            // Track memory after processing
            val memoryAfter = runtime.totalMemory() - runtime.freeMemory()
            val memoryUsed = memoryAfter - memoryBefore
            tracker.logMetric(TAG, "Memory used", "${memoryUsed / 1024} KB")

            val totalTime = tracker.stopTimer("total_ocr")
            tracker.logMetric(TAG, "Total OCR time", "${totalTime}ms")

            // Fix the FPS calculation - use Double for division
            val fps = if (totalTime > 0) 1000.0 / totalTime else 0.0
            tracker.logMetric(TAG, "FPS", "${String.format("%.2f", fps)}")

            // Display results
            runOnUiThread {
                txtResult.text = if (decodedText.isNotEmpty()) decodedText else "No text detected"
                txtAccuracy.text = "${String.format("%.0f", confidence)}%"
                btnPredict.isEnabled = true

                val message = if (decodedText.isNotEmpty()) {
                    "Prediction: $decodedText (${String.format("%.1f", confidence)}%) in ${totalTime}ms"
                } else {
                    "No text detected in the image"
                }
                Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
            }

        } catch (e: Exception) {
            Log.e(TAG, "OCR Error", e)
            runOnUiThread {
                Toast.makeText(this, "OCR Error: ${e.message}", Toast.LENGTH_LONG).show()
                txtResult.text = "Error: ${e.message}"
                txtAccuracy.text = "0%"
                btnPredict.isEnabled = true
            }
        }
    }

    private fun convertToGrayscale(bitmap: Bitmap): Bitmap {
        val startTime = System.currentTimeMillis()

        // Use ARGB_8888 for better performance
        val width = bitmap.width
        val height = bitmap.height
        val pixels = IntArray(width * height)

        // Get all pixels at once (FAST)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        // Process pixels in bulk
        for (i in pixels.indices) {
            val pixel = pixels[i]
            val r = (pixel shr 16) and 0xFF
            val g = (pixel shr 8) and 0xFF
            val b = pixel and 0xFF

            // Fast grayscale conversion
            val gray = (r * 0.299 + g * 0.587 + b * 0.114).toInt()
            pixels[i] = 0xFF000000.toInt() or (gray shl 16) or (gray shl 8) or gray
        }

        // Create bitmap from processed pixels
        val result = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        result.setPixels(pixels, 0, width, 0, 0, width, height)

        val time = System.currentTimeMillis() - startTime
        Log.d(TAG, "🔄 Grayscale conversion: ${time}ms")

        return result
    }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val buffer = ByteBuffer.allocateDirect(1 * IMAGE_HEIGHT * IMAGE_WIDTH * 1 * 4)
        buffer.order(ByteOrder.nativeOrder())

        for (y in 0 until IMAGE_HEIGHT) {
            for (x in 0 until IMAGE_WIDTH) {
                val pixel = bitmap.getPixel(x, y)
                val gray = Color.red(pixel)
                val normalized = gray / 255.0f
                buffer.putFloat(normalized)
            }
        }
        buffer.rewind()
        return buffer
    }

    private fun decodeCtcOutput(output: Array<FloatArray>): String {
        val decoded = StringBuilder()
        var lastIndex = -1

        for (t in 0 until TIME_STEPS) {
            var maxIndex = 0
            var maxProb = -1f

            for (c in 0 until NUM_CLASSES) {
                if (output[t][c] > maxProb) {
                    maxProb = output[t][c]
                    maxIndex = c
                }
            }

            if (maxIndex != BLANK_TOKEN && maxIndex != lastIndex) {
                if (maxIndex < CHARACTERS.length) {
                    decoded.append(CHARACTERS[maxIndex])
                    Log.d(TAG, "  Time $t: Added '${CHARACTERS[maxIndex]}' (prob=${String.format("%.3f", maxProb)})")
                }
                lastIndex = maxIndex
            } else if (maxIndex == BLANK_TOKEN) {
                lastIndex = -1
                Log.d(TAG, "  Time $t: Blank token (prob=${String.format("%.3f", maxProb)})")
            } else {
                Log.d(TAG, "  Time $t: Duplicate '${CHARACTERS[maxIndex]}', skipped")
            }
        }

        val result = decoded.toString().trim()
        Log.d(TAG, "Final decoded text: '$result' (${result.length} characters)")
        return result
    }

    private fun calculateConfidence(output: Array<FloatArray>): Float {
        var totalConfidence = 0f
        var count = 0

        for (t in 0 until TIME_STEPS) {
            val maxProb = output[t].maxOrNull() ?: 0f
            totalConfidence += maxProb
            count++
        }

        return if (count > 0) (totalConfidence / count) * 100 else 0f
    }
}