package com.example.glucoai

import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Android-side implementation of the preprocessing used by the
 * GlucoAI mobile training pipeline:
 *
 * raw camera PPG
 * -> high-pass motion removal
 * -> cardiac band-pass
 * -> smoothing
 * -> z-score normalization
 * -> peak detection
 * -> 11 features
 *
 * Note: the Python training implementation uses scipy filters. The
 * implementation below uses lightweight Kotlin filters so it can run
 * fully offline on Android.
 */
class PPGProcessor {

    companion object {
        private const val FPS = 30.0
        private const val N = 120
    }

    fun preprocessSignal(raw: FloatArray): FloatArray {
        require(raw.size == N) {
            "PPG signal must contain exactly 120 samples"
        }

        var x = raw.map { it.toDouble() }.toDoubleArray()

        // Remove low-frequency baseline / motion drift.
        x = highPass(x, 0.30, FPS)

        // Keep the cardiac band used by the Python pipeline: 0.7–3.5 Hz.
        x = bandPass(x, 0.70, 3.50, FPS)

        // Python uses Savitzky-Golay(11, 3). A short symmetric smoother
        // is used here to keep the Android implementation dependency-free.
        x = smooth(x, 11)

        // Exact Python normalization: (x - mean) / std.
        val mean = x.average()
        val variance = x.map { (it - mean) * (it - mean) }.average()
        val std = sqrt(variance)

        if (std < 1e-8) {
            return FloatArray(N)
        }

        return FloatArray(N) { i ->
            ((x[i] - mean) / std).toFloat()
        }
    }

    /**
     * The mobile model was trained with:
     * advanced 9 features + HRV 2 features.
     */
    fun extractFeatures(normalizedSignal: FloatArray): FloatArray {
        require(normalizedSignal.size == N)

        val mean = normalizedSignal.average().toFloat()

        var variance = 0.0
        for (v in normalizedSignal) {
            val d = v - mean
            variance += d * d
        }
        val std = sqrt(variance / N).toFloat()

        val maximum = normalizedSignal.maxOrNull() ?: 0f
        val minimum = normalizedSignal.minOrNull() ?: 0f
        val range = maximum - minimum

        val peaks = detectPeaks(normalizedSignal)

        var hrv = 0f
        var avgInterval = 0f

        if (peaks.size > 1) {
            val intervals = FloatArray(peaks.size - 1) { i ->
                (peaks[i + 1] - peaks[i]).toFloat()
            }

            avgInterval = intervals.average().toFloat()

            var s = 0.0
            for (v in intervals) {
                val d = v - avgInterval
                s += d * d
            }
            hrv = sqrt(s / intervals.size).toFloat()
        }

        var energy = 0.0
        for (v in normalizedSignal) {
            energy += v * v
        }

        val peakDensity = peaks.size.toFloat() / N

        // Python calculate_hrv_features:
        // SDNN = std(intervals)
        val sdnn = hrv

        // IMPORTANT:
        // The Python model uses sqrt(mean(intervals ** 2)),
        // not the usual successive-difference RMSSD.
        var rmssd = 0f
        if (peaks.size > 1) {
            var sumSquares = 0.0
            for (i in 1 until peaks.size) {
                val interval = (peaks[i] - peaks[i - 1]).toDouble()
                sumSquares += interval * interval
            }
            rmssd = sqrt(sumSquares / (peaks.size - 1)).toFloat()
        }

        return floatArrayOf(
            mean,
            std,
            maximum,
            minimum,
            hrv,
            avgInterval,
            energy.toFloat(),
            range,
            peakDensity,
            sdnn,
            rmssd
        )
    }

    /**
     * Peak detector approximating scipy.signal.find_peaks used by training:
     * minimum distance ~= 0.3 s at 30 FPS = 9 samples
     * prominence ~= 15% of signal range
     * height >= mean - std
     */
    fun detectPeaks(signal: FloatArray): IntArray {
        if (signal.size < 5) return IntArray(0)

        val mean = signal.average().toFloat()

        var variance = 0.0
        for (v in signal) {
            val d = v - mean
            variance += d * d
        }
        val std = sqrt(variance / signal.size).toFloat()

        val minValue = signal.minOrNull() ?: 0f
        val maxValue = signal.maxOrNull() ?: 0f
        val range = maxValue - minValue
        val minProminence = 0.15f * range
        val minDistance = 9

        val candidates = mutableListOf<Int>()

        for (i in 1 until signal.size - 1) {
            if (
                signal[i] > signal[i - 1] &&
                signal[i] >= signal[i + 1] &&
                signal[i] >= mean - std
            ) {
                val leftMin = maxOf(
                    0,
                    i - minDistance
                )
                val rightMax = minOf(
                    signal.lastIndex,
                    i + minDistance
                )

                var localMin = signal[i]
                for (j in leftMin..rightMax) {
                    if (signal[j] < localMin) {
                        localMin = signal[j]
                    }
                }

                if (signal[i] - localMin >= minProminence) {
                    candidates.add(i)
                }
            }
        }

        if (candidates.size >= 2) {
            return enforceMinimumDistance(signal, candidates, minDistance)
        }

        // Same relaxed fallback idea as the Python implementation.
        val relaxed = mutableListOf<Int>()
        for (i in 1 until signal.size - 1) {
            if (
                signal[i] > signal[i - 1] &&
                signal[i] >= signal[i + 1]
            ) {
                relaxed.add(i)
            }
        }

        return enforceMinimumDistance(signal, relaxed, 5)
    }

    fun isSignalUsable(signal: FloatArray): Boolean {
        if (signal.size != N) return false

        val min = signal.minOrNull() ?: return false
        val max = signal.maxOrNull() ?: return false

        return (max - min) > 1e-5f
    }

    private fun enforceMinimumDistance(
        signal: FloatArray,
        candidates: List<Int>,
        minDistance: Int
    ): IntArray {
        if (candidates.isEmpty()) return IntArray(0)

        val selected = mutableListOf<Int>()

        for (index in candidates) {
            if (selected.isEmpty()) {
                selected.add(index)
                continue
            }

            val last = selected.last()

            if (index - last >= minDistance) {
                selected.add(index)
            } else if (signal[index] > signal[last]) {
                selected[selected.lastIndex] = index
            }
        }

        return selected.toIntArray()
    }

    // Simple first-order high-pass.
    private fun highPass(
        input: DoubleArray,
        cutoffHz: Double,
        fs: Double
    ): DoubleArray {
        if (input.size < 3) return input

        val rc = 1.0 / (2.0 * Math.PI * cutoffHz)
        val dt = 1.0 / fs
        val alpha = rc / (rc + dt)

        val output = DoubleArray(input.size)
        output[0] = 0.0

        for (i in 1 until input.size) {
            output[i] =
                alpha * (
                        output[i - 1] +
                                input[i] -
                                input[i - 1]
                        )
        }

        return output
    }

    // Cascaded first-order low/high pass approximation of the 0.7–3.5 Hz band.
    private fun bandPass(
        input: DoubleArray,
        lowHz: Double,
        highHz: Double,
        fs: Double
    ): DoubleArray {
        val highPassed = highPass(input, lowHz, fs)
        return lowPass(highPassed, highHz, fs)
    }

    private fun lowPass(
        input: DoubleArray,
        cutoffHz: Double,
        fs: Double
    ): DoubleArray {
        if (input.isEmpty()) return input

        val rc = 1.0 / (2.0 * Math.PI * cutoffHz)
        val dt = 1.0 / fs
        val alpha = dt / (rc + dt)

        val output = DoubleArray(input.size)
        output[0] = input[0]

        for (i in 1 until input.size) {
            output[i] =
                output[i - 1] +
                        alpha * (input[i] - output[i - 1])
        }

        return output
    }

    private fun smooth(
        input: DoubleArray,
        window: Int
    ): DoubleArray {
        if (input.size < 3) return input

        val radius = window / 2
        val output = DoubleArray(input.size)

        for (i in input.indices) {
            var sum = 0.0
            var count = 0

            val start = maxOf(0, i - radius)
            val end = minOf(input.lastIndex, i + radius)

            for (j in start..end) {
                sum += input[j]
                count++
            }

            output[i] = sum / count
        }

        return output
    }
}
