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
        try {
            Log.d(TAG, "STEP 1: Starting OCR processing")

            // Show processing message
            runOnUiThread {
                txtResult.text = "Processing..."
                txtAccuracy.text = "0%"
                btnPredict.isEnabled = false
            }

            // Preprocess image
            val grayBitmap = convertToGrayscale(bitmap)
            val resized = Bitmap.createScaledBitmap(grayBitmap, IMAGE_WIDTH, IMAGE_HEIGHT, true)
            val inputBuffer = convertBitmapToByteBuffer(resized)

            // Create output buffer
            val output = Array(1) { Array(TIME_STEPS) { FloatArray(NUM_CLASSES) } }

            // Run inference
            tflite.run(inputBuffer, output)
            Log.d(TAG, "STEP 2: Inference completed")

            // Decode CTC output
            val decodedText = decodeCtcOutput(output[0])
            Log.d(TAG, "STEP 3: Decoded text: '$decodedText' (${decodedText.length} chars)")

            // Calculate confidence
            val confidence = calculateConfidence(output[0])

            // Display results
            runOnUiThread {
                txtResult.text = if (decodedText.isNotEmpty()) decodedText else "No text detected"
                txtAccuracy.text = "${String.format("%.0f", confidence)}%"
                btnPredict.isEnabled = true

                val message = if (decodedText.isNotEmpty()) {
                    "Prediction completed: $decodedText (${String.format("%.1f", confidence)}%)"
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
        val width = bitmap.width
        val height = bitmap.height
        val grayBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)

        for (x in 0 until width) {
            for (y in 0 until height) {
                val pixel = bitmap.getPixel(x, y)
                val r = Color.red(pixel)
                val g = Color.green(pixel)
                val b = Color.blue(pixel)
                val gray = (0.299 * r + 0.587 * g + 0.114 * b).toInt()
                grayBitmap.setPixel(x, y, Color.rgb(gray, gray, gray))
            }
        }
        return grayBitmap
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