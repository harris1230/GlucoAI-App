package com.example.glucoai

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.view.ViewGroup
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Divider
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.example.glucoai.ui.theme.GlucoAITheme
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.max
import kotlin.math.min

class MainActivity : ComponentActivity() {

    private lateinit var cameraExecutor: ExecutorService
    private var camera: Camera? = null
    private var imageAnalysis: ImageAnalysis? = null
    private var previewView: PreviewView? = null

    private val ppgSamples = mutableListOf<Float>()

    @Volatile
    private var isCollecting = false

    private var predictionReady = false

    @Volatile
    private var currentSignal = 0f

    @Volatile
    private var minimumSignal = 0f

    @Volatile
    private var maximumSignal = 0f

    var cameraReady by mutableStateOf(false)
        private set

    private lateinit var ppgProcessor: PPGProcessor
    private lateinit var glucosePredictor: GlucosePredictor

    private val cameraPermissionLauncher =
        registerForActivityResult(
            ActivityResultContracts.RequestPermission()
        ) { granted ->
            if (granted) {
                previewView?.let { connectPreview(it) }
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        cameraExecutor = Executors.newSingleThreadExecutor()
        ppgProcessor = PPGProcessor()
        glucosePredictor = GlucosePredictor(this)

        setContent {
            GlucoAITheme {
                GlucoAIScreen()
            }
        }
    }

    // =========================================================
    // CAMERA
    // =========================================================

    private fun connectPreview(preview: PreviewView) {

        if (
            ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
            return
        }

        val cameraProviderFuture =
            ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({

            try {
                val cameraProvider = cameraProviderFuture.get()

                val cameraPreview = Preview.Builder()
                    .build()

                cameraPreview.setSurfaceProvider(
                    preview.surfaceProvider
                )

                imageAnalysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(
                        ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST
                    )
                    .build()

                imageAnalysis?.setAnalyzer(
                    cameraExecutor
                ) { imageProxy ->
                    processFrame(imageProxy)
                }

                cameraProvider.unbindAll()

                camera = cameraProvider.bindToLifecycle(
                    this,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    cameraPreview,
                    imageAnalysis
                )

                cameraReady = true

            } catch (e: Exception) {
                cameraReady = false
                e.printStackTrace()
            }

        }, ContextCompat.getMainExecutor(this))
    }

    // =========================================================
    // CAMERA FRAME PROCESSING
    // =========================================================

    private fun processFrame(imageProxy: ImageProxy) {

        try {

            if (!isCollecting) {
                imageProxy.close()
                return
            }

            val greenIntensity =
                calculateGreenIntensity(imageProxy)

            if (greenIntensity.isFinite()) {

                synchronized(ppgSamples) {

                    if (ppgSamples.size < 120) {

                        ppgSamples.add(
                            greenIntensity.toFloat()
                        )

                        currentSignal =
                            greenIntensity.toFloat()

                        if (ppgSamples.size == 1) {

                            minimumSignal =
                                greenIntensity.toFloat()

                            maximumSignal =
                                greenIntensity.toFloat()

                        } else {

                            minimumSignal =
                                min(
                                    minimumSignal,
                                    greenIntensity.toFloat()
                                )

                            maximumSignal =
                                max(
                                    maximumSignal,
                                    greenIntensity.toFloat()
                                )
                        }

                        if (ppgSamples.size >= 120) {

                            isCollecting = false
                            predictionReady = true

                            camera?.cameraControl
                                ?.enableTorch(false)
                        }
                    }
                }
            }

        } catch (e: Exception) {
            e.printStackTrace()
        } finally {
            imageProxy.close()
        }
    }

    // =========================================================
    // Green CHANNEL EXTRACTION
    // =========================================================

    private fun calculateGreenIntensity(
        image: ImageProxy
    ): Double {

        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]

        val yBuffer = yPlane.buffer
        val uBuffer = uPlane.buffer
        val vBuffer = vPlane.buffer

        val width = image.width
        val height = image.height

        val yRowStride = yPlane.rowStride
        val yPixelStride = yPlane.pixelStride

        val uRowStride = uPlane.rowStride
        val uPixelStride = uPlane.pixelStride

        val vRowStride = vPlane.rowStride
        val vPixelStride = vPlane.pixelStride

        val startX = (width * 0.25f).toInt()
        val endX = (width * 0.75f).toInt()

        val startY = (height * 0.25f).toInt()
        val endY = (height * 0.75f).toInt()

        var greenSum = 0.0
        var count = 0

        val stepX = max(1, (endX - startX) / 25)
        val stepY = max(1, (endY - startY) / 25)

        for (y in startY until endY step stepY) {

            for (x in startX until endX step stepX) {

                val yIndex =
                    y * yRowStride +
                            x * yPixelStride

                val chromaX = x / 2
                val chromaY = y / 2

                val uIndex =
                    chromaY * uRowStride +
                            chromaX * uPixelStride

                val vIndex =
                    chromaY * vRowStride +
                            chromaX * vPixelStride

                if (
                    yIndex !in 0 until yBuffer.limit() ||
                    uIndex !in 0 until uBuffer.limit() ||
                    vIndex !in 0 until vBuffer.limit()
                ) {
                    continue
                }

                val yValue =
                    yBuffer.get(yIndex).toInt() and 0xFF

                val uValue =
                    uBuffer.get(uIndex).toInt() and 0xFF

                val vValue =
                    vBuffer.get(vIndex).toInt() and 0xFF

                // YUV -> approximate GREEN
                val green =
                    yValue -
                            0.344136 *
                            (uValue - 128) -
                            0.714136 *
                            (vValue - 128)

                greenSum +=
                    green.coerceIn(0.0, 255.0)

                count++
            }
        }

        if (count == 0) {
            return 0.0
        }

        return greenSum / count
    }

    // =========================================================
    // START PPG
    // =========================================================

    fun startPPG(): Boolean {

        if (camera == null || !cameraReady) {
            return false
        }

        synchronized(ppgSamples) {

            ppgSamples.clear()

            currentSignal = 0f
            minimumSignal = 0f
            maximumSignal = 0f

            predictionReady = false
            isCollecting = true
        }

        try {
            camera?.cameraControl?.enableTorch(true)
        } catch (e: Exception) {
            e.printStackTrace()
        }

        return true
    }

    // =========================================================
    // STOP PPG
    // =========================================================

    fun stopPPG() {

        isCollecting = false

        try {
            camera?.cameraControl?.enableTorch(false)
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    // =========================================================
    // GET SAMPLES
    // =========================================================

    fun getCurrentSamples(): FloatArray {

        synchronized(ppgSamples) {
            return ppgSamples.toFloatArray()
        }
    }

    // =========================================================
    // PREDICTION
    // =========================================================

    fun preparePrediction(
        onResult: (Float) -> Unit,
        onError: (String) -> Unit
    ) {

        val samples = getCurrentSamples()

        if (samples.size != 120) {
            onError(
                "Please collect exactly 120 PPG samples first."
            )
            return
        }

        try {

            if (!ppgProcessor.isSignalUsable(samples)) {

                onError(
                    "PPG signal is not usable. Please try again."
                )
                return
            }

            val glucose =
                glucosePredictor.predict(
                    samples,
                    ppgProcessor
                )

            onResult(glucose)

            onResult(glucose)

        } catch (e: Exception) {

            e.printStackTrace()

            onError(
                e.message ?: "Prediction failed."
            )
        }
    }

    // =========================================================
    // UI
    // =========================================================

    @Composable
    private fun GlucoAIScreen() {

        var sampleCount by remember {
            mutableIntStateOf(0)
        }

        var signal by remember {
            mutableFloatStateOf(0f)
        }

        var minSignal by remember {
            mutableFloatStateOf(0f)
        }

        var maxSignal by remember {
            mutableFloatStateOf(0f)
        }

        var prediction by remember {
            mutableStateOf<Float?>(null)
        }

        var status by remember {
            mutableStateOf(
                "Allow camera access to begin."
            )
        }

        var collecting by remember {
            mutableStateOf(false)
        }

        LaunchedEffect(Unit) {

            while (true) {

                kotlinx.coroutines.delay(200)

                synchronized(ppgSamples) {

                    sampleCount = ppgSamples.size
                    signal = currentSignal
                    minSignal = minimumSignal
                    maxSignal = maximumSignal
                    collecting = isCollecting
                }

                if (cameraReady && sampleCount == 0 && !collecting) {
                    status =
                        "Place your finger over the rear camera and flash."
                }

                if (sampleCount >= 120) {
                    status =
                        "PPG captured successfully. Ready for prediction."
                }
            }
        }

        val progress =
            (sampleCount / 120f).coerceIn(0f, 1f)

        Column(
            modifier = Modifier
                .fillMaxSize()
                .background(
                    Brush.verticalGradient(
                        listOf(
                            Color(0xFF0B1020),
                            Color(0xFF111827),
                            Color(0xFF080B14)
                        )
                    )
                )
                .verticalScroll(rememberScrollState())
                .padding(18.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {

            // HEADER

            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically
            ) {

                Box(
                    modifier = Modifier
                        .size(46.dp)
                        .clip(CircleShape)
                        .background(Color(0xFF263BFF)),
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        text = "♥",
                        color = Color.White,
                        fontSize = 22.sp,
                        fontWeight = FontWeight.Bold
                    )
                }

                Spacer(Modifier.width(12.dp))

                Column(
                    modifier = Modifier.weight(1f)
                ) {
                    Text(
                        text = "GlucoAI",
                        color = Color.White,
                        fontSize = 26.sp,
                        fontWeight = FontWeight.Bold
                    )

                    Text(
                        text = "Camera-based PPG prototype",
                        color = Color(0xFF9CA3AF),
                        fontSize = 12.sp
                    )
                }

                Box(
                    modifier = Modifier
                        .clip(RoundedCornerShape(50))
                        .background(
                            if (cameraReady)
                                Color(0xFF123D2A)
                            else
                                Color(0xFF3A2530)
                        )
                        .padding(
                            horizontal = 10.dp,
                            vertical = 6.dp
                        )
                ) {
                    Text(
                        text =
                            if (cameraReady) "● READY"
                            else "● CAMERA",
                        color =
                            if (cameraReady)
                                Color(0xFF5EEB9B)
                            else
                                Color(0xFFFF9AAE),
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
            }

            Spacer(Modifier.height(18.dp))

            // CAMERA CARD

            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(24.dp),
                colors = CardDefaults.cardColors(
                    containerColor = Color(0xFF171D2D)
                )
            ) {

                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(330.dp)
                        .clip(RoundedCornerShape(24.dp))
                ) {

                    AndroidView(
                        factory = { context ->

                            PreviewView(context).apply {

                                previewView = this

                                layoutParams =
                                    ViewGroup.LayoutParams(
                                        ViewGroup.LayoutParams.MATCH_PARENT,
                                        ViewGroup.LayoutParams.MATCH_PARENT
                                    )

                                scaleType =
                                    PreviewView.ScaleType.FILL_CENTER

                                post {
                                    connectPreview(this)
                                }
                            }
                        },

                        modifier = Modifier.fillMaxSize()
                    )

                    // CAMERA TOP LABEL

                    Surface(
                        modifier = Modifier
                            .align(Alignment.TopStart)
                            .padding(14.dp),
                        shape = RoundedCornerShape(50),
                        color = Color.Black.copy(alpha = 0.65f)
                    ) {

                        Row(
                            modifier = Modifier.padding(
                                horizontal = 10.dp,
                                vertical = 7.dp
                            ),
                            verticalAlignment =
                                Alignment.CenterVertically
                        ) {

                            Text(
                                text = "●",
                                color = Color.White,
                                fontSize = 12.sp
                            )

                            Spacer(Modifier.width(5.dp))

                            Text(
                                text = "LIVE CAMERA",
                                color = Color.White,
                                fontSize = 10.sp,
                                fontWeight = FontWeight.Bold
                            )
                        }
                    }

                    // FINGER GUIDE

                    if (!collecting && sampleCount == 0) {

                        Column(
                            modifier = Modifier
                                .align(Alignment.Center)
                                .clip(RoundedCornerShape(18.dp))
                                .background(
                                    Color.Black.copy(alpha = 0.55f)
                                )
                                .padding(18.dp),
                            horizontalAlignment =
                                Alignment.CenterHorizontally
                        ) {

                            Text(
                                text = "♥",
                                color = Color.White,
                                fontSize = 28.sp,
                                fontWeight = FontWeight.Bold
                            )

                            Spacer(Modifier.height(8.dp))

                            Text(
                                text = "Place finger on camera",
                                color = Color.White,
                                fontWeight = FontWeight.Bold
                            )

                            Text(
                                text = "Cover the rear camera + flash",
                                color = Color(0xFFD1D5DB),
                                fontSize = 11.sp
                            )
                        }
                    }

                    // COLLECTING INDICATOR

                    if (collecting) {

                        Surface(
                            modifier = Modifier
                                .align(Alignment.TopEnd)
                                .padding(14.dp),
                            shape = RoundedCornerShape(50),
                            color = Color(0xFF7F1D1D)
                        ) {

                            Row(
                                modifier = Modifier.padding(
                                    horizontal = 10.dp,
                                    vertical = 7.dp
                                ),
                                verticalAlignment =
                                    Alignment.CenterVertically
                            ) {

                                CircularProgressIndicator(
                                    modifier = Modifier.size(12.dp),
                                    strokeWidth = 2.dp,
                                    color = Color.White
                                )

                                Spacer(Modifier.width(6.dp))

                                Text(
                                    text = "CAPTURING",
                                    color = Color.White,
                                    fontSize = 10.sp,
                                    fontWeight = FontWeight.Bold
                                )
                            }
                        }
                    }
                }
            }

            Spacer(Modifier.height(14.dp))

            // STATUS

            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(18.dp),
                colors = CardDefaults.cardColors(
                    containerColor = Color(0xFF151A29)
                )
            ) {

                Row(
                    modifier = Modifier.padding(14.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {

                    Text(
                        text = if (sampleCount >= 120) "✓" else "i",
                        color =
                            if (sampleCount >= 120)
                                Color(0xFF5EEB9B)
                            else
                                Color(0xFF8FA2FF),
                        fontSize = 22.sp,
                        fontWeight = FontWeight.Bold
                    )

                    Spacer(Modifier.width(10.dp))

                    Text(
                        text = status,
                        color = Color(0xFFE5E7EB),
                        fontSize = 13.sp
                    )
                }
            }

            Spacer(Modifier.height(14.dp))

            // PPG PROGRESS

            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(20.dp),
                colors = CardDefaults.cardColors(
                    containerColor = Color(0xFF151A29)
                )
            ) {

                Column(
                    modifier = Modifier.padding(16.dp)
                ) {

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement =
                            Arrangement.SpaceBetween
                    ) {

                        Text(
                            text = "PPG Capture",
                            color = Color.White,
                            fontWeight = FontWeight.Bold
                        )

                        Text(
                            text = "$sampleCount / 120",
                            color = Color(0xFF9CA3AF),
                            fontSize = 12.sp
                        )
                    }

                    Spacer(Modifier.height(10.dp))

                    LinearProgressIndicator(
                        progress = { progress },
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(8.dp)
                            .clip(RoundedCornerShape(50)),
                        color = Color(0xFF6375FF),
                        trackColor = Color(0xFF293044)
                    )

                    Spacer(Modifier.height(14.dp))

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement =
                            Arrangement.SpaceBetween
                    ) {

                        SignalMetric(
                            title = "CURRENT",
                            value = "%.2f".format(signal)
                        )

                        SignalMetric(
                            title = "MIN",
                            value = "%.2f".format(minSignal)
                        )

                        SignalMetric(
                            title = "MAX",
                            value = "%.2f".format(maxSignal)
                        )
                    }
                }
            }

            Spacer(Modifier.height(14.dp))

            // ACTION BUTTONS

            Button(
                onClick = {

                    prediction = null

                    if (startPPG()) {
                        status =
                            "Capturing PPG... Keep your finger still."
                    } else {
                        status =
                            "Camera is not ready yet."
                    }
                },
                enabled = cameraReady && !collecting,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(54.dp),
                shape = RoundedCornerShape(17.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = Color(0xFF596BFF),
                    disabledContainerColor = Color(0xFF30364A)
                )
            ) {

                Text(
                    text = "▶",
                    fontSize = 15.sp
                )

                Spacer(Modifier.width(8.dp))

                Text(
                    text = "START PPG",
                    fontWeight = FontWeight.Bold
                )
            }

            Spacer(Modifier.height(9.dp))

            OutlinedButton(
                onClick = {

                    stopPPG()

                    status =
                        "PPG collection stopped."

                },
                enabled = collecting,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(52.dp),
                shape = RoundedCornerShape(17.dp)
            ) {

                Text(
                    text = "■",
                    fontSize = 13.sp
                )

                Spacer(Modifier.width(8.dp))

                Text("STOP")
            }

            Spacer(Modifier.height(9.dp))

            Button(
                onClick = {

                    preparePrediction(

                        onResult = { glucose ->

                            prediction = glucose

                            status =
                                "Prediction completed successfully."

                        },

                        onError = { error ->
                            status = error
                        }
                    )

                },
                enabled = sampleCount == 120 && !collecting,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(54.dp),
                shape = RoundedCornerShape(17.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = Color(0xFF19A974),
                    disabledContainerColor = Color(0xFF303A3A)
                )
            ) {

                Text(
                    text = "♥",
                    fontSize = 16.sp
                )

                Spacer(Modifier.width(8.dp))

                Text(
                    text = "CALCULATE GLUCOSE",
                    fontWeight = FontWeight.Bold
                )
            }

            Spacer(Modifier.height(18.dp))

            // RESULT

            if (prediction != null) {

                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(26.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = Color(0xFF172A28)
                    )
                ) {

                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(24.dp),
                        horizontalAlignment =
                            Alignment.CenterHorizontally
                    ) {

                        Text(
                            text = "ESTIMATED GLUCOSE",
                            color = Color(0xFF86EFAC),
                            fontSize = 12.sp,
                            fontWeight = FontWeight.Bold
                        )

                        Spacer(Modifier.height(8.dp))

                        Text(
                            text =
                                "%.1f".format(prediction),
                            color = Color.White,
                            fontSize = 44.sp,
                            fontWeight = FontWeight.Bold
                        )

                        Text(
                            text = "mg/dL",
                            color = Color(0xFFB7C5C1),
                            fontSize = 15.sp
                        )

                        Spacer(Modifier.height(14.dp))

                        Divider(
                            color = Color(0xFF2C4A44)
                        )

                        Spacer(Modifier.height(12.dp))

                        Text(
                            text =
                                "Prototype estimate • not a medical diagnosis",
                            color = Color(0xFF9CA3AF),
                            fontSize = 10.sp,
                            textAlign = TextAlign.Center
                        )
                    }
                }

            } else {

                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(22.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = Color(0xFF141927)
                    )
                ) {

                    Column(
                        modifier = Modifier.padding(22.dp),
                        horizontalAlignment =
                            Alignment.CenterHorizontally
                    ) {

                        Text(
                            text = "♥",
                            color = Color(0xFF596BFF),
                            fontSize = 28.sp,
                            fontWeight = FontWeight.Bold
                        )

                        Spacer(Modifier.height(8.dp))

                        Text(
                            text = "Estimated Glucose",
                            color = Color.White,
                            fontSize = 15.sp,
                            fontWeight = FontWeight.Bold
                        )

                        Spacer(Modifier.height(4.dp))

                        Text(
                            text = "-- mg/dL",
                            color = Color(0xFF6B7280),
                            fontSize = 25.sp,
                            fontWeight = FontWeight.Bold
                        )
                    }
                }
            }

            Spacer(Modifier.height(18.dp))

            // HOW IT WORKS

            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(20.dp),
                colors = CardDefaults.cardColors(
                    containerColor = Color(0xFF141927)
                )
            ) {

                Column(
                    modifier = Modifier.padding(17.dp)
                ) {

                    Text(
                        text = "How it works",
                        color = Color.White,
                        fontWeight = FontWeight.Bold
                    )

                    Spacer(Modifier.height(10.dp))

                    InfoRow(
                        icon = "01",
                        title = "Cover camera",
                        description =
                            "Place your finger over the rear camera and flash."
                    )

                    InfoRow(
                        icon = "02",
                        title = "Capture PPG",
                        description =
                            "GlucoAI collects 120 camera-based signal samples."
                    )

                    InfoRow(
                        icon = "03",
                        title = "Run model",
                        description =
                            "The local ML model estimates glucose from the signal."
                    )
                }
            }

            Spacer(Modifier.height(16.dp))

            Row(
                verticalAlignment = Alignment.CenterVertically
            ) {

                Text(
                    text = "⚡",
                    color = Color(0xFFFFD166),
                    fontSize = 12.sp
                )

                Spacer(Modifier.width(5.dp))

                Text(
                    text =
                        "Keep your finger still during measurement",
                    color = Color(0xFF9CA3AF),
                    fontSize = 10.sp
                )
            }

            Spacer(Modifier.height(8.dp))

            Text(
                text = "GlucoAI • Camera-based PPG prototype",
                color = Color(0xFF4B5563),
                fontSize = 10.sp
            )

            Spacer(Modifier.height(10.dp))
        }
    }

    @Composable
    private fun SignalMetric(
        title: String,
        value: String
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                text = title,
                color = Color(0xFF6B7280),
                fontSize = 9.sp,
                fontWeight = FontWeight.Bold
            )

            Spacer(Modifier.height(2.dp))

            Text(
                text = value,
                color = Color.White,
                fontSize = 13.sp,
                fontWeight = FontWeight.SemiBold
            )
        }
    }

    @Composable
    private fun InfoRow(
        icon: String,
        title: String,
        description: String
    ) {

        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 6.dp),
            verticalAlignment = Alignment.Top
        ) {

            Box(
                modifier = Modifier
                    .size(30.dp)
                    .clip(CircleShape)
                    .background(Color(0xFF252C45)),
                contentAlignment = Alignment.Center
            ) {

                Text(
                    text = icon,
                    color = Color(0xFF91A0FF),
                    fontSize = 9.sp,
                    fontWeight = FontWeight.Bold
                )
            }

            Spacer(Modifier.width(10.dp))

            Column {

                Text(
                    text = title,
                    color = Color.White,
                    fontSize = 12.sp,
                    fontWeight = FontWeight.Bold
                )

                Text(
                    text = description,
                    color = Color(0xFF8F98A8),
                    fontSize = 10.sp
                )
            }
        }
    }

    // =========================================================
    // CLEANUP
    // =========================================================

    override fun onDestroy() {

        isCollecting = false

        try {
            camera?.cameraControl?.enableTorch(false)
        } catch (_: Exception) {
        }

        imageAnalysis?.clearAnalyzer()
        cameraExecutor.shutdown()
        glucosePredictor.close()

        super.onDestroy()
    }
}
