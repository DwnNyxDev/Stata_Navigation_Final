package com.example.statanavigation

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.core.content.ContextCompat.startActivity
import com.example.statanavigation.databinding.ActivityMainBinding
import java.nio.ByteBuffer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors


import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage

import androidx.core.graphics.scale
import org.json.JSONArray
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import org.pytorch.torchvision.TensorImageUtils
import org.w3c.dom.Text
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import kotlin.math.exp

private lateinit var module: Module
private lateinit var classNames: List<String>
private lateinit var textImageClass: TextView


class CameraActivity : AppCompatActivity() {
    private lateinit var viewBinding: ActivityMainBinding

    private lateinit var cameraExecutor: ExecutorService

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        viewBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)

        textImageClass = findViewById(R.id.textView2)
        textImageClass.setText("")

        // Request camera permissions
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            requestPermissions()
        }

        cameraExecutor = Executors.newSingleThreadExecutor()

        val modelFilePath: String = assetFilePath(this, "model.ptl")
        module = LiteModuleLoader.load(modelFilePath)
        classNames = loadClassNames(this, "model_classes.json")
//
    }

    private val activityResultLauncher =
        registerForActivityResult(
            ActivityResultContracts.RequestMultiplePermissions())
        { permissions ->
            // Handle Permission granted/rejected
            var permissionGranted = true
            permissions.entries.forEach {
                if (it.key in REQUIRED_PERMISSIONS && it.value == false)
                    permissionGranted = false
            }
            if (!permissionGranted) {
                Toast.makeText(baseContext,
                    "Permission request denied",
                    Toast.LENGTH_SHORT).show()
            } else {
                startCamera()
            }
        }

    private class LocationAnalyzer(
        private val context: Context,
    ) : ImageAnalysis.Analyzer {

        private var lastUpdate : Long = 0

        private fun ByteBuffer.toByteArray(): ByteArray {
            rewind()    // Rewind the buffer to zero
            val data = ByteArray(remaining())
            get(data)   // Copy the buffer into a byte array
            return data // Return the byte array
        }

        override fun analyze(image: ImageProxy) {
            if (System.currentTimeMillis() > lastUpdate + 1000) {
                lastUpdate = System.currentTimeMillis()

                var bitmap = imageProxyToBitmap(image)

                //scale image down to 256x256
                bitmap = bitmap.scale(256, 256)

                //center crop image to 224 x 224
                val x = (bitmap.getWidth() - 224) / 2
                val y = (bitmap.getHeight() - 224) / 2

                bitmap = Bitmap.createBitmap(bitmap, x, y, 224, 224)

                //Standard ImageNet means and stds
                val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
                val std = floatArrayOf(0.229f, 0.224f, 0.225f)

                // Convert bitmap to normalized tensor bc model must take in a Tensor
                val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap, mean, std)

                //Pass image through model
                val outputTensor = module.forward(IValue.from(inputTensor)).toTensor()

                val scores = outputTensor.dataAsFloatArray

                //convert scores to probabilities using softmax function
                val probs = softmax(scores)

                //find index of highest probability
                var maxScore = -Float.MAX_VALUE
                var maxScoreIdx = -1
                for (i in 0..<probs.size) {
                    if (probs[i] > maxScore) {
                        maxScore = probs[i]
                        maxScoreIdx = i
                    }
                }

                //Find highest probable class using index of highest probability

                val className: String = classNames[maxScoreIdx]
                val confidence = (maxScore * 100).toInt()

                val activity = context as AppCompatActivity
                activity.runOnUiThread {
                    //set text to highest probable class
                    val message = "$className\n Confidence: $confidence%"
                    textImageClass.setText(message)
                }
            }

            image.close()
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            // Used to bind the lifecycle of cameras to the lifecycle owner
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            // Preview
            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(viewBinding.viewFinder.surfaceProvider)
                }

            val imageAnalyzer = ImageAnalysis.Builder()
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor, LocationAnalyzer(this, ))
                }

            // Select back camera as a default
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                // Unbind use cases before rebinding
                cameraProvider.unbindAll()

                // Bind use cases to camera
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageAnalyzer)

            } catch(exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))


    }

    private fun requestPermissions() {
        activityResultLauncher.launch(REQUIRED_PERMISSIONS)
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
            baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    companion object {
        private const val TAG = "CameraXApp"
        private const val FILENAME_FORMAT = "yyyy-MM-dd-HH-mm-ss-SSS"
        private val REQUIRED_PERMISSIONS =
            mutableListOf (
                Manifest.permission.CAMERA,
            ).toTypedArray()
    }
}

fun imageProxyToBitmap(image: ImageProxy): Bitmap {
    val yBuffer = image.planes[0].buffer
    val uBuffer = image.planes[1].buffer
    val vBuffer = image.planes[2].buffer

    val ySize = yBuffer.remaining()
    val uSize = uBuffer.remaining()
    val vSize = vBuffer.remaining()

    val nv21 = ByteArray(ySize + uSize + vSize)

    yBuffer.get(nv21, 0, ySize)
    vBuffer.get(nv21, ySize, vSize)
    uBuffer.get(nv21, ySize + vSize, uSize)

    val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
    val out = ByteArrayOutputStream()
    yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 100, out)
    val imageBytes = out.toByteArray()

    return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
}

//Converts logits into probabilities using softmax function
fun softmax(logits: FloatArray): FloatArray {
    var max = Float.NEGATIVE_INFINITY
    for (`val` in logits) {
        if (`val` > max) max = `val`
    }

    var sum = 0.0
    val exps = DoubleArray(logits.size)
    for (i in logits.indices) {
        exps[i] = exp((logits[i] - max).toDouble())
        sum += exps[i]
    }

    val softmax = FloatArray(logits.size)
    for (i in logits.indices) {
        softmax[i] = (exps[i] / sum).toFloat()
    }
    return softmax
}

//Load in class names from json file
fun loadClassNames(context: Context, fileName: String): List<String> {
    val inputStream = context.assets.open(fileName)
    val jsonStr = inputStream.bufferedReader().use { it.readText() }
    val jsonArray = JSONArray(jsonStr)

    val classNames = mutableListOf<String>()
    for (i in 0 until jsonArray.length()) {
        classNames.add(jsonArray.getString(i))
    }
    return classNames
}

// Copies the asset file to a real path and returns the File object
@Throws(IOException::class)
fun assetFilePath(context: Context, assetName: String): String {
    val file = File(context.filesDir, assetName)
    if (file.exists() && file.length() > 0) {
        return file.absolutePath
    }

    val assetManager = context.assets
    assetManager.open(assetName).use { `is` ->
        FileOutputStream(file).use { os ->
            val buffer = ByteArray(4096)
            var read: Int
            while ((`is`.read(buffer).also { read = it }) != -1) {
                os.write(buffer, 0, read)
            }
            os.flush()
        }
    }
    return file.absolutePath
}