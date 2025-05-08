package com.example.statanavigation

import android.content.Intent
import android.os.Bundle
import androidx.activity.ComponentActivity


class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val cameraIntent = Intent (
            this,
            CameraActivity::class.java
        )

        startActivity(cameraIntent)
    }
}