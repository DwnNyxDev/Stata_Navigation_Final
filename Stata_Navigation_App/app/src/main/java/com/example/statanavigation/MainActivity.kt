package com.example.statanavigation

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.graphics.scale
import org.pytorch.IValue
//import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.util.Arrays
import kotlin.math.exp


class MainActivity : ComponentActivity() {
    private lateinit var selectImageButton: Button
    private lateinit var previewImageView: ImageView
    private lateinit var textImageClass: TextView

    private lateinit var imagePickerLauncher: ActivityResultLauncher<String>

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        //set screen contents as variables
        selectImageButton = findViewById(R.id.BSelectImage)
        previewImageView = findViewById(R.id.IVPreviewImage)
        textImageClass = findViewById(R.id.textView2)

        //initialize text
        textImageClass.setText("No Image Loaded")

        // Register the ActivityResultLauncher
        imagePickerLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
            uri?.let {
                //set imageview to be selected image
                previewImageView.setImageURI(it)
                val `is` = contentResolver.openInputStream(uri!!)

                //load stata-trained model from assets folder
                val modelFilePath: String = assetFilePath(this, "epoch_10_scripted.pt")
                val module = Module.load(modelFilePath)

                //convert image to bitmap and preprocess it
                var bitmap = BitmapFactory.decodeStream(`is`)

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
                for (i in 0..< probs.size) {
                    if (probs[i] > maxScore) {
                        maxScore = probs[i]
                        maxScoreIdx = i
                    }
                }

                //Find highest probable class using index of highest probability
                val classes = listOf("4th floor elevator", "4th floor entrance", "4th floor patio", "4th floor r&d pub", "4th floor stata")
                val className: String = classes[maxScoreIdx]

                //set text to highest probable class
                textImageClass.setText(className)

                `is`!!.close()
            }


        }

        // Handle the button click
        selectImageButton.setOnClickListener {
            imagePickerLauncher.launch("image/*")
        }
    }
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