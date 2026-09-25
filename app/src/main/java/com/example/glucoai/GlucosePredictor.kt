package com.example.glucoai

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.min

class GlucosePredictor(context: Context) {

    private val interpreter: Interpreter

    init {
        val model = context.assets
            .open("glucoai_mobile.tflite")
            .readBytes()

        val buffer = ByteBuffer.allocateDirect(model.size)
        buffer.order(ByteOrder.nativeOrder())
        buffer.put(model)
        buffer.rewind()

        interpreter = Interpreter(buffer)

        Log.d(
            "GlucoAI",
            "TFLite input count=${interpreter.inputTensorCount}"
        )

        for (i in 0 until interpreter.inputTensorCount) {
            val tensor = interpreter.getInputTensor(i)
            Log.d(
                "GlucoAI",
                "input[$i] shape=${tensor.shape().contentToString()} type=${tensor.dataType()}"
            )
        }

        val out = interpreter.getOutputTensor(0)
        Log.d(
            "GlucoAI",
            "output shape=${out.shape().contentToString()} type=${out.dataType()}"
        )
    }

    fun predict(
        rawSignal: FloatArray,
        processor: PPGProcessor
    ): Float {

        require(rawSignal.size == 120) {
            "Signal must contain exactly 120 values"
        }

        // IMPORTANT:
        // Training applies preprocessing BEFORE both the signal input
        // and feature extraction.
        val processedSignal =
            processor.preprocessSignal(rawSignal)

        val features =
            processor.extractFeatures(processedSignal)

        return predictProcessed(
            processedSignal,
            features
        )
    }

    private fun predictProcessed(
        signal: FloatArray,
        features: FloatArray
    ): Float {

        // Same StandardScaler statistics used during training.
        val featureMean = floatArrayOf(
            -0.0076283297f,
            0.7474608608f,
            1.6444253623f,
            -1.5015850017f,
            4.2820590610f,
            26.6708335082f,
            82.1231613557f,
            3.1460103591f,
            0.0388888900f,
            4.2820590610f,
            27.2455271085f
        )

        val featureScale = floatArrayOf(
            0.0330070134f,
            0.3528658325f,
            0.7371463516f,
            0.7206447057f,
            4.5897797209f,
            8.5616612109f,
            70.4915858439f,
            1.3856508101f,
            0.0184256935f,
            4.5897797209f,
            9.0398438799f
        )

        val scaledFeatures = Array(1) {
            FloatArray(11)
        }

        for (i in 0 until 11) {
            scaledFeatures[0][i] =
                (features[i] - featureMean[i]) /
                        featureScale[i]
        }

        val inputSignal = Array(1) {
            Array(120) {
                FloatArray(1)
            }
        }

        for (i in 0 until 120) {
            inputSignal[0][i][0] = signal[i]
        }

        val output = Array(1) {
            FloatArray(1)
        }

        interpreter.runForMultipleInputsOutputs(
            arrayOf(
                inputSignal,
                scaledFeatures
            ),
            mapOf(0 to output)
        )

        val scaledPrediction = output[0][0]

        Log.d(
            "GlucoAI",
            "processed signal min=${signal.minOrNull()} max=${signal.maxOrNull()} mean=${signal.average()}"
        )
        Log.d(
            "GlucoAI",
            "features=${features.contentToString()}"
        )
        Log.d(
            "GlucoAI",
            "scaledFeatures=${scaledFeatures[0].contentToString()}"
        )
        Log.d(
            "GlucoAI",
            "TFLite scaled prediction=$scaledPrediction"
        )

        // MinMaxScaler inverse transform.
        val glucose =
            (scaledPrediction + 0.730769217f) /
                    0.0076923077f

        // Keep UI from showing absurd values if the model produces
        // an out-of-distribution result.
        val safeGlucose =
            glucose.coerceIn(40f, 400f)

        Log.d(
            "GlucoAI",
            "final glucose=$safeGlucose mg/dL"
        )

        return safeGlucose
    }

    fun close() {
        interpreter.close()
    }
}
