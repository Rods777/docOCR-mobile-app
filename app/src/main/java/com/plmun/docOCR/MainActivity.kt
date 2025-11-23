package com.plmun.docOCR

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.res.AssetFileDescriptor
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.enableEdgeToEdge
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class MainActivity : ComponentActivity() {

    private lateinit var imgPreview: ImageView
    private lateinit var txtUpload: TextView
    private lateinit var txtCapture: TextView
    private lateinit var txtResult: TextView
    private lateinit var tflite: Interpreter

    companion object {
        private const val TAG = "MainActivity"
        private const val IMAGE_WIDTH = 160
        private const val IMAGE_HEIGHT = 64

        // Characters exactly as in Python: string.ascii_letters + string.digits + " -'.,:"
        private val CHARACTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -'.,:"

        private const val BLANK_INDEX = 0  // Blank token is at index 0
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)

        imgPreview = findViewById(R.id.img_preview)
        txtUpload = findViewById(R.id.txt_upload)
        txtCapture = findViewById(R.id.txt_capture)
        txtResult = findViewById(R.id.txt_result)

        // Initialize TensorFlow Lite Interpreter
        try {
            tflite = Interpreter(loadModelFile())
            Log.d(TAG, "✅ TensorFlow Lite Interpreter loaded successfully")
        } catch (e: Exception) {
            Log.e(TAG, "❌ Failed to load TensorFlow Lite model", e)
            Toast.makeText(this, "Failed to load OCR model", Toast.LENGTH_LONG).show()
        }

        txtUpload.setOnClickListener {
            val intent = Intent(Intent.ACTION_PICK)
            intent.type = "image/*"
            galleryLauncher.launch(intent)
        }

        txtCapture.setOnClickListener {
            requestCameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        tflite.close()
    }

    private fun loadModelFile(): MappedByteBuffer {
        val fileDescriptor: AssetFileDescriptor =
            assets.openFd("ml/ocr_model_production_fp16.tflite")
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private val requestCameraPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted ->
            if (isGranted) openCamera()
            else Toast.makeText(this, "Camera permission denied", Toast.LENGTH_SHORT).show()
        }

    private fun openCamera() {
        val cameraIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        cameraLauncher.launch(cameraIntent)
    }

    // GALLERY IMAGE
    private val galleryLauncher =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            if (result.resultCode == Activity.RESULT_OK) {
                val uri = result.data?.data
                imgPreview.setImageURI(uri)

                try {
                    val stream = contentResolver.openInputStream(uri!!)
                    val bitmap = BitmapFactory.decodeStream(stream)
                    stream?.close()

                    runOcr(bitmap)
                } catch (e: Exception) {
                    Log.e(TAG, "Error loading image from gallery", e)
                    Toast.makeText(this, "Error loading image", Toast.LENGTH_SHORT).show()
                }
            }
        }

    // CAMERA IMAGE
    private val cameraLauncher =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            if (result.resultCode == Activity.RESULT_OK) {
                val bitmap = result.data?.extras?.get("data") as? Bitmap
                if (bitmap != null) {
                    imgPreview.setImageBitmap(bitmap)
                    runOcr(bitmap)
                } else {
                    Toast.makeText(this, "Error capturing image", Toast.LENGTH_SHORT).show()
                }
            }
        }

    // ----------------------------
    // 🔥 RUN OCR MODEL HERE (USING RAW INTERPRETER)
    // ----------------------------

    private fun runOcr(bitmap: Bitmap) {
        try {
            Log.d(TAG, "STEP 1: runOcr started")

            val resized = Bitmap.createScaledBitmap(bitmap, IMAGE_WIDTH, IMAGE_HEIGHT, true)
            Log.d(TAG, "STEP 2: Bitmap resized: ${resized.width}x${resized.height}")

            val inputBuffer = convertBitmapToByteBuffer(resized)
            Log.d(TAG, "STEP 3: Input buffer size: ${inputBuffer.capacity()} bytes")

            // Get model input and output details
            val inputShape = tflite.getInputTensor(0).shape()
            Log.d(TAG, "STEP 4: Model input shape: ${inputShape.contentToString()}")

            // 🔥 FIX: Force the correct output shape based on your Python model
            // Your model should output [1, 40, 69] (time_steps = 40, num_classes = 69)
            val expectedOutputShape = intArrayOf(1, 40, 69)
            val outputBuffer = TensorBuffer.createFixedSize(expectedOutputShape, DataType.FLOAT32)

            Log.d(
                TAG,
                "STEP 5: Using forced output shape: ${expectedOutputShape.contentToString()}"
            )
            Log.d(TAG, "STEP 6: Output buffer size: ${outputBuffer.flatSize} elements")
            Log.d(TAG, "STEP 7: Output buffer bytes: ${outputBuffer.flatSize * 4} bytes")

            // 🔥 FIX: Resize the interpreter output tensor
            tflite.resizeInput(0, intArrayOf(1, IMAGE_HEIGHT, IMAGE_WIDTH, 1))
            tflite.allocateTensors()

            // Run inference
            Log.d(TAG, "STEP 8: Running inference...")
            tflite.run(inputBuffer, outputBuffer.buffer.rewind())
            Log.d(TAG, "STEP 9: Inference completed successfully")

            val ocrText = decodeOutput(outputBuffer, expectedOutputShape)
            Log.d(TAG, "STEP 10: Decoded text = $ocrText")

            txtResult.text = "Result: $ocrText"

        } catch (e: Exception) {
            Log.e(TAG, "OCR Error", e)
            runOnUiThread {
                Toast.makeText(this, "OCR Error: ${e.message}", Toast.LENGTH_LONG).show()
                txtResult.text = "Error: ${e.message}"
            }
        }
    }

    // Convert Bitmap → ByteBuffer
    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val buffer = ByteBuffer.allocateDirect(1 * IMAGE_HEIGHT * IMAGE_WIDTH * 1 * 4)
        buffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(IMAGE_HEIGHT * IMAGE_WIDTH)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        for (pixel in pixels) {
            // This matches your Python: cv2.imread(..., IMREAD_GRAYSCALE)
            val r = (pixel shr 16 and 0xFF)
            val g = (pixel shr 8 and 0xFF)
            val b = (pixel and 0xFF)

            // This matches OpenCV's grayscale conversion
            val gray = (0.299f * r + 0.587f * g + 0.114f * b)

            // This matches your Python: img / 255.0
            val normalized = gray / 255.0f

            buffer.putFloat(normalized)
        }
        buffer.rewind()
        return buffer
    }

    // FLEXIBLE CTC DECODING
    private fun decodeOutput(tensor: TensorBuffer, shape: IntArray): String {
        val floatArray = tensor.floatArray

        Log.d(TAG, "DEBUG: Output shape = ${shape.contentToString()}")
        Log.d(TAG, "DEBUG: Output elements = ${floatArray.size}")

        // Handle the expected shape [1, 40, 69]
        if (shape.size == 3 && shape[0] == 1 && shape[1] == 40 && shape[2] == 69) {
            return decodeCtcOutput3D(floatArray, 40, 69)
        } else {
            return "Unexpected shape: ${shape.contentToString()}"
        }
    }

    private fun decodeCtcOutput3D(floatArray: FloatArray, timeSteps: Int, numClasses: Int): String {
        val decodedText = StringBuilder()
        var lastIndex = -1

        Log.d(TAG, "DEBUG: Decoding $timeSteps time steps with $numClasses classes")

        // Iterate through each time step
        for (timeStep in 0 until timeSteps) {
            var maxIndex = -1
            var maxValue = Float.MIN_VALUE

            // Find the character with highest probability at this time step
            for (charIndex in 0 until numClasses) {
                val arrayIndex = timeStep * numClasses + charIndex
                if (arrayIndex < floatArray.size && floatArray[arrayIndex] > maxValue) {
                    maxValue = floatArray[arrayIndex]
                    maxIndex = charIndex
                }
            }

            // CTC decoding: remove blanks and consecutive duplicates
            if (maxIndex != BLANK_INDEX && maxIndex != lastIndex) {
                // Convert character index to actual character
                val charPos = maxIndex - 1
                if (charPos >= 0 && charPos < CHARACTERS.length) {
                    decodedText.append(CHARACTERS[charPos])
                    Log.d(
                        TAG,
                        "DEBUG: Time step $timeStep -> char '${CHARACTERS[charPos]}' (index: $maxIndex)"
                    )
                } else if (maxIndex > 0) {
                    Log.w(TAG, "Character index out of bounds: $charPos")
                }
            }
            lastIndex = maxIndex
        }

        val result = decodedText.toString()
        Log.d(TAG, "DEBUG: Final decoded text = '$result'")
        return result.ifEmpty { "No text detected" }
    }
}