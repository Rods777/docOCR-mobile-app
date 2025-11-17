package com.plmun.docOCR

import android.Manifest
import android.app.Activity
import android.content.Intent
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
import com.plmun.docOCR.ml.OcrModelProductionFp16
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import org.tensorflow.lite.DataType
import java.nio.ByteBuffer
import java.nio.ByteOrder

class MainActivity : ComponentActivity() {

    private lateinit var imgPreview: ImageView
    private lateinit var txtUpload: TextView
    private lateinit var txtCapture: TextView
    private lateinit var txtResult: TextView

    companion object {
        private const val TAG = "MainActivity"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)

        imgPreview = findViewById(R.id.img_preview)
        txtUpload = findViewById(R.id.txt_upload)
        txtCapture = findViewById(R.id.txt_capture)
        txtResult = findViewById(R.id.txt_result)

        txtUpload.setOnClickListener {
            val intent = Intent(Intent.ACTION_PICK)
            intent.type = "image/*"
            galleryLauncher.launch(intent)
        }

        txtCapture.setOnClickListener {
            requestCameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
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
    // 🔥 RUN OCR MODEL HERE
    // ----------------------------

    private fun runOcr(bitmap: Bitmap) {
        try {
            val resized = Bitmap.createScaledBitmap(bitmap, 160, 64, true)
            val byteBuffer = convertBitmapToByteBuffer(resized)

            val model = OcrModelProductionFp16.newInstance(this)

            // Create input tensor
            val input = TensorBuffer.createFixedSize(intArrayOf(1, 64, 160, 1), DataType.FLOAT32)
            input.loadBuffer(byteBuffer)

            // Process the model
            val outputs = model.process(input)

            // Get the output tensor - handle the larger output size (11040 bytes = 2760 floats)
            val outputTensor = outputs.outputFeature0AsTensorBuffer

            // The error suggests output should be 11040 bytes = 2760 float values (11040 / 4)
            val result = outputTensor.floatArray

            Log.d(TAG, "Output tensor size: ${result.size} floats, ${result.size * 4} bytes")

            model.close()

            // Convert output tensor → text
            val ocrText = decodeOutput(result)
            txtResult.text = ocrText

        } catch (e: Exception) {
            e.printStackTrace()
            Toast.makeText(this, "OCR Error: ${e.message}", Toast.LENGTH_LONG).show()
            txtResult.text = "Error: ${e.message}"
        }
    }

    // Convert Bitmap → ByteBuffer
    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val buffer = ByteBuffer.allocateDirect(1 * 64 * 160 * 1 * 4) // FLOAT32 = 4 bytes
        buffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(64 * 160)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        for (pixel in pixels) {
            // Convert to grayscale
            val r = (pixel shr 16 and 0xFF) / 255.0f
            val g = (pixel shr 8 and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f
            val gray = (0.299f * r + 0.587f * g + 0.114f * b)
            buffer.putFloat(gray)
        }
        buffer.rewind()
        return buffer
    }

    // MODEL-SPECIFIC DECODING - Updated for CTC output
    private fun decodeOutput(arr: FloatArray): String {
        if (arr.isEmpty()) return "No output"

        // For CTC models, the output is typically [batch_size, time_steps, num_classes]
        // Your output has 2760 floats, which might be something like [1, 69, 40] or similar

        Log.d(TAG, "Output array size: ${arr.size}")

        // Simple greedy decoding for CTC output
        val characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "

        // Assuming the output shape is [1, time_steps, num_classes]
        // We need to find the time_steps and num_classes
        val numClasses = characters.length + 1 // +1 for CTC blank
        val timeSteps = arr.size / numClasses

        Log.d(TAG, "Assuming timeSteps: $timeSteps, numClasses: $numClasses")

        val decodedText = StringBuilder()
        var prevIndex = -1

        for (t in 0 until timeSteps) {
            var maxProb = -1.0f
            var maxIndex = -1

            // Find the character with highest probability at this time step
            for (c in 0 until numClasses) {
                val prob = arr[t * numClasses + c]
                if (prob > maxProb) {
                    maxProb = prob
                    maxIndex = c
                }
            }

            // CTC decoding: skip blanks and repeated characters
            if (maxIndex != numClasses - 1 && maxIndex != prevIndex) { // numClasses-1 is usually blank
                if (maxIndex < characters.length) {
                    decodedText.append(characters[maxIndex])
                }
            }
            prevIndex = maxIndex
        }

        return if (decodedText.isNotEmpty()) decodedText.toString() else "No text detected"
    }
}