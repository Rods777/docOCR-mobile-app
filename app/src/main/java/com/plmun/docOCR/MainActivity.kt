package com.plmun.docOCR

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.res.AssetFileDescriptor
import android.graphics.Bitmap
import android.graphics.BitmapFactory
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
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

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

        // 78 medical terms (from your Python list)
        private val MEDICAL_TERMS = arrayOf(
            "Aceta", "Ace", "Alatrol", "Amodis", "Atrizin", "Axodin", "Azithrocin",
            "Azyth", "Az", "Bacaid", "Backtone", "Baclofen", "Baclon", "Bacmax",
            "Beklo", "Bicozin", "Canazole", "Candinil", "Cetisoft", "Conaz", "Dancel",
            "Denixil", "Diflu", "Dinafex", "Disopan", "Esonix", "Esoral", "Etizin",
            "Exium", "Fenadin", "Fexofast", "Fexo", "Filmet", "Fixal", "Flamyd",
            "Flexibac", "Flexilax", "Flugal", "Ketocon", "Ketoral", "Ketotab",
            "Ketozol", "Leptic", "Lucan-R", "Lumona", "M-Kast", "Maxima", "Maxpro",
            "Metro", "Metsina", "Monas", "Montair", "Montene", "Montex", "Napa Extend",
            "Napa", "Nexcap", "Nexum", "Nidazyl", "Nizoder", "Odmon", "Omastin",
            "Opton", "Progut", "Provair", "Renova", "Rhinil", "Ritch", "Rivotril",
            "Romycin", "Rozith", "Sergel", "Tamen", "Telfast", "Tridosil", "Trilock",
            "Vifas", "Zithrin"
        )

        // Fuzzy-match threshold (0.0 - 1.0). If no close match above threshold, fall back to raw predicted token.
        private const val FUZZY_THRESHOLD = 0.35
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)

        imgPreview = findViewById(R.id.img_preview)
        txtUpload = findViewById(R.id.txt_upload)
        txtCapture = findViewById(R.id.txt_capture)
        txtResult = findViewById(R.id.txt_result)

        // Load TFLite model
        try {
            tflite = Interpreter(loadModelFile())
            Log.d(TAG, "✅ TensorFlow Lite Interpreter loaded successfully")

            // Log input and output tensor shapes & datatypes
            val inTensor = tflite.getInputTensor(0)
            val outTensor = tflite.getOutputTensor(0)

            Log.d(TAG, "📊 TFLite input shape: ${inTensor.shape().contentToString()} dtype=${inTensor.dataType()}")
            Log.d(TAG, "📊 TFLite output shape: ${outTensor.shape().contentToString()} dtype=${outTensor.dataType()} bytes=${outTensor.numBytes()}")

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
        if (::tflite.isInitialized) tflite.close()
    }

    private fun loadModelFile(): MappedByteBuffer {
        val fileDescriptor: AssetFileDescriptor = assets.openFd("ml/CNN_BiLSTM_v2fp16.tflite")
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
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

    private val cameraLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == Activity.RESULT_OK) {
            val bmp = result.data?.extras?.get("data") as? Bitmap
            if (bmp != null) {
                imgPreview.setImageBitmap(bmp)
                runOcr(bmp)
            } else {
                Toast.makeText(this, "Error capturing image", Toast.LENGTH_SHORT).show()
            }
        }
    }

    // ----------------------------
    // RUN INFERENCE: classification-ready for your model
    // ----------------------------
    private fun runOcr(bitmap: Bitmap) {
        try {
            Log.d(TAG, "STEP 1: runOcr started")

            // Resize bitmap to model input
            val resized = Bitmap.createScaledBitmap(bitmap, IMAGE_WIDTH, IMAGE_HEIGHT, true)
            Log.d(TAG, "STEP 2: Bitmap resized: ${resized.width}x${resized.height}")

            // Convert to ByteBuffer
            val inputBuffer = convertBitmapToByteBuffer(resized)
            Log.d(TAG, "STEP 3: Input buffer size: ${inputBuffer.capacity()} bytes")

            // Get output tensor info
            val outputTensor = tflite.getOutputTensor(0)
            val outputShape = outputTensor.shape()
            val numClasses = if (outputShape.size >= 2) outputShape[1] else 0

            if (numClasses > 0) {
                // Allocate Float32 output buffer
                val outputBuffer = ByteBuffer.allocateDirect(numClasses * 4)
                outputBuffer.order(ByteOrder.nativeOrder())

                // Run inference
                tflite.run(inputBuffer, outputBuffer)
                outputBuffer.rewind()

                // Read floats
                val outputFloats = FloatArray(numClasses)
                for (i in 0 until numClasses) outputFloats[i] = outputBuffer.float

                // Get top 3 indices
                val top3 = outputFloats
                    .mapIndexed { index, value -> index to value }
                    .sortedByDescending { it.second }
                    .take(3)

                // Prepare result string
                val resultText = top3.joinToString(separator = "\n") { (index, value) ->
                    val term = if (index in MEDICAL_TERMS.indices) MEDICAL_TERMS[index] else "Unknown"
                    "Prediction: $term | Confidence: ${"%.4f".format(value)}"
                }

                txtResult.text = resultText
                return
            }

            txtResult.text = "Unexpected model output shape: ${outputShape.contentToString()}"
            Log.e(TAG, "Unexpected model output shape: ${outputShape.contentToString()}")

        } catch (e: Exception) {
            Log.e(TAG, "OCR Error", e)
            runOnUiThread {
                Toast.makeText(this, "OCR Error: ${e.message}", Toast.LENGTH_LONG).show()
                txtResult.text = "Error: ${e.message}"
            }
        }
    }


    // Convert a bitmap (already resized to IMAGE_WIDTH x IMAGE_HEIGHT) into a ByteBuffer that matches your Python preprocessing:
    // 1) convert to grayscale [0..1]
    // 2) compute per-image mean/std
    // 3) apply z-score normalization (value - mean) / std
    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        // Expect bitmap size equals IMAGE_WIDTH x IMAGE_HEIGHT
        val width = bitmap.width
        val height = bitmap.height

        // Allocate buffer for float32 (1 x H x W x 1)
        val buffer = ByteBuffer.allocateDirect(1 * IMAGE_HEIGHT * IMAGE_WIDTH * 1 * 4)
        buffer.order(ByteOrder.nativeOrder())

        val grayValues = FloatArray(width * height)
        var p = 0
        for (y in 0 until height) {
            for (x in 0 until width) {
                val pixel = bitmap.getPixel(x, y)
                val r = (pixel shr 16) and 0xFF
                val g = (pixel shr 8) and 0xFF
                val b = pixel and 0xFF
                val gray = (0.299f * r + 0.587f * g + 0.114f * b)
                grayValues[p++] = gray / 255f
            }
        }

        // mean and std (per-image)
        val mean = grayValues.average().toFloat()
        val variance = if (grayValues.isNotEmpty()) grayValues.map { (it - mean) * (it - mean) }.average().toFloat() else 0f
        val std = sqrt(variance)
        val stdSafe = if (std < 1e-6f) 1e-6f else std

        // write normalized floats (row-major)
        p = 0
        for (y in 0 until height) {
            for (x in 0 until width) {
                val normalized = (grayValues[p++] - mean) / stdSafe
                buffer.putFloat(normalized)
            }
        }
        buffer.rewind()
        return buffer
    }

    // ---------- Fuzzy matching utilities (pure Kotlin) ----------

    // Return the closest term from MEDICAL_TERMS to `word` using normalized Levenshtein similarity
    private fun findClosestTerm(word: String): String {
        var bestTerm = word
        var bestScore = -1.0
        for (term in MEDICAL_TERMS) {
            val score = similarityScore(word, term)
            if (score > bestScore) {
                bestScore = score
                bestTerm = term
            }
        }
        return bestTerm
    }

    // Normalized similarity score in [0..1] based on Levenshtein distance
    private fun similarityScore(a: String, b: String): Double {
        if (a.isEmpty() && b.isEmpty()) return 1.0
        if (a.isEmpty() || b.isEmpty()) return 0.0
        val dist = levenshtein(a.lowercase(), b.lowercase())
        val maxLen = max(a.length, b.length)
        return 1.0 - (dist.toDouble() / maxLen.toDouble())
    }

    // Levenshtein distance (iterative DP)
    private fun levenshtein(s: String, t: String): Int {
        val n = s.length
        val m = t.length
        if (n == 0) return m
        if (m == 0) return n

        val v0 = IntArray(m + 1) { it }    // previous row
        val v1 = IntArray(m + 1)          // current row

        for (i in 0 until n) {
            v1[0] = i + 1
            val si = s[i]
            for (j in 0 until m) {
                val cost = if (si == t[j]) 0 else 1
                v1[j + 1] = min(min(v1[j] + 1, v0[j + 1] + 1), v0[j] + cost)
            }
            // copy v1 to v0
            for (j in 0..m) v0[j] = v1[j]
        }
        return v1[m]
    }

    // Placeholder for your previous CTC decode if you ever switch to a sequence model
    private fun decodeCtcOutput3D(floatArray: FloatArray, timeSteps: Int, numClasses: Int): String {
        // Basic greedy decoding similar to your earlier implementation
        val decoded = StringBuilder()
        var lastIndex = -1
        for (t in 0 until timeSteps) {
            var maxIdx = -1
            var maxVal = Float.MIN_VALUE
            for (c in 0 until numClasses) {
                val idx = t * numClasses + c
                if (idx < floatArray.size && floatArray[idx] > maxVal) {
                    maxVal = floatArray[idx]
                    maxIdx = c
                }
            }
            if (maxIdx != -1 && maxIdx != lastIndex && maxIdx > 0) {
                // Note: this assumes a separate mapping - not used for current classifier model
                decoded.append("?") // replace with actual char mapping if you change model
            }
            lastIndex = maxIdx
        }
        return decoded.toString()
    }
}
