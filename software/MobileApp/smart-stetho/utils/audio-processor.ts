// utils/audio-processor.ts

/**
 * Bandpass Filter (20Hz - 500Hz)
 * Optimized for heart sounds at 4000Hz sampling rate.
 * Removes low-frequency 'rumble' and high-frequency 'hiss'.
 */
export const applyHeartFilter = (data: Int16Array): Int16Array => {
  const n = data.length
  const filtered = new Int16Array(n)

  // Internal state for IIR filters
  let lowPassState = 0
  let highPassState = 0

  // Alpha coefficients (Adjusted for 4000Hz)
  // LP Alpha 0.45 targets roughly 500Hz
  // HP Alpha 0.98 targets roughly 20Hz
  const alphaLP = 0.45
  const alphaHP = 0.98

  for (let i = 0; i < n; i++) {
    // 1. Low Pass Filter (Removes high-freq electronic noise)
    lowPassState = alphaLP * data[i] + (1 - alphaLP) * lowPassState

    // 2. High Pass Filter (Removes DC offset and motion artifacts)
    const delta = lowPassState - (i > 0 ? data[i - 1] : data[i])
    highPassState = alphaHP * (highPassState + delta)

    // Clamp to Int16 range to prevent clipping
    filtered[i] = Math.max(-32768, Math.min(32767, highPassState))
  }

  return filtered
}
