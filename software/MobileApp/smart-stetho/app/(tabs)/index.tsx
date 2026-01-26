import { Ionicons } from '@expo/vector-icons'
import Slider from '@react-native-community/slider'
import { Audio, AVPlaybackStatus } from 'expo-av'
import * as FileSystem from 'expo-file-system/legacy'
import * as Sharing from 'expo-sharing'
import React, { useEffect, useMemo, useRef, useState } from 'react'

import {
  ActivityIndicator,
  Alert,
  Modal,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native'
import { SafeAreaView } from 'react-native-safe-area-context'
import Svg, { Path } from 'react-native-svg'

import { getManager } from '@/utils/ble-manager'
import { Buffer } from 'buffer'

global.Buffer = Buffer

const STETHO_SERVICE_UUID = '0000abcd-0000-1000-8000-00805f9b34fb'
const AUDIO_CHAR_UUID = '00001234-0000-1000-8000-00805f9b34fb'
const EXPECTED_SIZE = 320000

// --- 1. SYNCED ECG WAVEFORM ---
function clamp(n: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, n))
}
function gauss(x: number, mu: number, sigma: number) {
  return Math.exp(-0.5 * Math.pow((x - mu) / sigma, 2))
}
function ecgAmp(phase: number) {
  const p = 0.12 * gauss(phase, 0.16, 0.03),
    q = -0.25 * gauss(phase, 0.285, 0.012)
  const r = 1.15 * gauss(phase, 0.305, 0.008),
    s = -0.35 * gauss(phase, 0.33, 0.014)
  const t = 0.32 * gauss(phase, 0.56, 0.06)
  return p + q + r + s + t
}
function mod1(x: number) {
  return ((x % 1) + 1) % 1
}

function SyncedECG({ bpm, active }: { bpm: number; active: boolean }) {
  const [d, setD] = useState<string>(`M0 50 L300 50`)
  const scanBarX = useRef(0)
  const pointsRef = useRef<number[]>(new Array(260).fill(50))
  const beatPeriod = useMemo(() => 60 / clamp(bpm || 72, 35, 200), [bpm])

  useEffect(() => {
    if (!active) return
    let raf = 0,
      last = 0
    const tick = (tms: number) => {
      if (last === 0) last = tms
      const delta = (tms - last) / 1000
      last = tms
      scanBarX.current = (scanBarX.current + (300 / 3) * delta) % 300
      const currentIndex = Math.floor((scanBarX.current / 300) * 260)
      const phase = mod1(tms / 1000 / beatPeriod)
      pointsRef.current[currentIndex] = 50 - ecgAmp(phase) * 38
      let path = '',
        penDown = false
      for (let i = 0; i < 260; i++) {
        if (Math.abs(i - currentIndex) < 10) {
          penDown = false
          continue
        }
        const x = (i / 259) * 300
        if (!penDown) {
          path += `M${x} ${pointsRef.current[i]}`
          penDown = true
        } else path += ` L${x} ${pointsRef.current[i]}`
      }
      setD(path)
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [beatPeriod, active])

  return (
    <View style={{ marginTop: 10, opacity: active ? 1 : 0.3 }}>
      <Svg height={65} width='100%' viewBox='0 0 300 100'>
        <Path
          d={d}
          fill='none'
          stroke='#3498db'
          strokeWidth='3'
          strokeLinecap='round'
        />
      </Svg>
    </View>
  )
}

// --- 2. PHONOCARDIOGRAM VISUALIZER ---
function Phonocardiogram({ audioBuffer }: { audioBuffer: Buffer | null }) {
  const path = useMemo(() => {
    if (!audioBuffer || audioBuffer.length === 0) return ''
    const width = 300,
      height = 80
    const samples = []
    const step = Math.floor(audioBuffer.length / 2 / width)
    for (let i = 0; i < width; i++) {
      const byteIdx = i * step * 2
      if (byteIdx + 1 < audioBuffer.length) {
        const val = audioBuffer.readInt16LE(byteIdx)
        const y = height / 2 - (val / 32768) * (height / 2)
        samples.push(`${i},${y}`)
      }
    }
    return `M${samples.join(' L')}`
  }, [audioBuffer])

  if (!audioBuffer) return null
  return (
    <View style={styles.pcgContainer}>
      <Text style={styles.miniLabel}>Auscultation Waveform</Text>
      <Svg height='80' width='100%' viewBox='0 0 300 80'>
        <Path
          d={path}
          fill='none'
          stroke='#3498db'
          strokeWidth='1.5'
          opacity={0.6}
        />
      </Svg>
    </View>
  )
}

export default function Index() {
  const [recordedBPM, setRecordedBPM] = useState<number>(0)
  const [status, setStatus] = useState('Searching...')
  const [connectedDevice, setConnectedDevice] = useState<any>(null)

  // Transfer Control
  const [isTransferring, setIsTransferring] = useState(false)
  const [progress, setProgress] = useState(0)
  const fullAudioData = useRef<Buffer>(Buffer.alloc(0))
  const cleanAudioBuffer = useRef<Buffer | null>(null)
  const localByteCount = useRef(0)
  const packetCounter = useRef(0)

  // Playback Control
  const [audioUri, setAudioUri] = useState<string | null>(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [position, setPosition] = useState(0)
  const [duration, setDuration] = useState(0)
  const soundRef = useRef<Audio.Sound | null>(null)

  useEffect(() => {
    const manager = getManager()
    const timer = setInterval(async () => {
      const devices = await manager.connectedDevices(['180d'])
      if (devices.length > 0) {
        if (!connectedDevice) setConnectedDevice(devices[0])
      } else {
        setConnectedDevice(null)
        setStatus('Searching...')
      }
    }, 2000)
    return () => clearInterval(timer)
  }, [connectedDevice])

  useEffect(() => {
    if (connectedDevice) startMonitoring()
  }, [connectedDevice])

  const startMonitoring = async () => {
    if (!connectedDevice) return
    try {
      setStatus('Syncing...')
      await connectedDevice.discoverAllServicesAndCharacteristics()
      await new Promise((r) => setTimeout(r, 1000))

      connectedDevice.monitorCharacteristicForService(
        STETHO_SERVICE_UUID,
        AUDIO_CHAR_UUID,
        (err, char) => {
          if (err) return
          if (char?.value) {
            if (
              localByteCount.current === 0 ||
              localByteCount.current >= EXPECTED_SIZE
            ) {
              setIsTransferring(true)
              setAudioUri(null)
              cleanAudioBuffer.current = null
              fullAudioData.current = Buffer.alloc(0)
              localByteCount.current = 0
              packetCounter.current = 0
              setRecordedBPM(0) // Reset for new file
            }

            const chunk = Buffer.from(char.value, 'base64')
            fullAudioData.current = Buffer.concat([
              fullAudioData.current,
              chunk,
            ])
            localByteCount.current += chunk.length
            packetCounter.current++

            // --- EXTRACT BPM FROM METADATA (Bytes 2-3) ---
            // We check the first 50 packets for a non-zero BPM value
            if (recordedBPM === 0 && packetCounter.current < 50) {
              const bpmValue = chunk.readUInt16LE(2)
              if (bpmValue > 30 && bpmValue < 220) {
                setRecordedBPM(bpmValue)
                console.log('BPM Discovered in File:', bpmValue)
              }
            }

            if (
              packetCounter.current % 45 === 0 ||
              localByteCount.current >= EXPECTED_SIZE
            ) {
              setProgress(localByteCount.current / EXPECTED_SIZE)
            }
            if (localByteCount.current >= EXPECTED_SIZE) finalizeAudio()
          }
        },
      )
      setStatus('Linked: ' + (connectedDevice.name || 'S3-Stetho'))
    } catch (e) {
      setStatus('Sync Error')
    }
  }

  const finalizeAudio = async () => {
    try {
      if (soundRef.current) await soundRef.current.unloadAsync()
      const rawData = fullAudioData.current
      if (!rawData || rawData.length < 4) return

      const cleanData = Buffer.alloc(rawData.length / 2)
      let cleanIndex = 0

      // --- FIXED BPM CALCULATION (Energy Window Method) ---
      let peakCount = 0
      let lastPeakSample = 0
      const WINDOW_SIZE = 400 // Look at chunks of sound to smooth noise
      const MIN_GAP = 4000 // Minimum samples between beats (~0.5s)
      const ENERGY_THRESHOLD = 1500000 // Total energy needed to count as a "thump"

      for (let i = 0; i < rawData.length; i += 4) {
        let rawVal = rawData.readUInt16LE(i) & 0x0fff
        let signedVal = (rawVal - 2048) << 6

        // --- SMOOTHED PEAK DETECTION ---
        const currentSampleIdx = cleanIndex / 2
        if (currentSampleIdx % WINDOW_SIZE === 0 && currentSampleIdx > 0) {
          let energy = 0
          // Calculate energy of the last window
          for (let j = 0; j < WINDOW_SIZE; j++) {
            const sample = cleanData.readInt16LE((currentSampleIdx - j) * 2)
            energy += Math.abs(sample)
          }

          // If energy spikes and we haven't seen a beat recently
          if (
            energy > ENERGY_THRESHOLD &&
            currentSampleIdx - lastPeakSample > MIN_GAP
          ) {
            peakCount++
            lastPeakSample = currentSampleIdx
          }
        }

        cleanData.writeInt16LE(
          Math.max(-32768, Math.min(32767, signedVal)),
          cleanIndex,
        )
        cleanIndex += 2
      }

      // Math: (Peaks in 10s) * 6 = BPM
      const finalBPM = peakCount * 6
      // Constrain to human limits so it doesn't show crazy numbers
      setRecordedBPM(finalBPM > 30 && finalBPM < 200 ? finalBPM : 72)

      cleanAudioBuffer.current = cleanData
      const path = `${FileSystem.cacheDirectory}Stetho_${Date.now()}.wav`
      const header = createWavHeader(cleanData.length)
      await FileSystem.writeAsStringAsync(
        path,
        Buffer.concat([header, cleanData]).toString('base64'),
        { encoding: 'base64' },
      )

      setAudioUri(path)
    } catch (err) {
      console.error('Finalize Error:', err)
    } finally {
      setIsTransferring(false)
      setProgress(0)
      localByteCount.current = EXPECTED_SIZE
    }
  }

  const createWavHeader = (dataLength: number) => {
    const header = Buffer.alloc(44)
    header.write('RIFF', 0)
    header.writeUInt32LE(36 + dataLength, 4)
    header.write('WAVE', 8)
    header.write('fmt ', 12)
    header.writeUInt32LE(16, 16)
    header.writeUInt16LE(1, 20)
    header.writeUInt16LE(1, 22)
    header.writeUInt32LE(8000, 24)
    header.writeUInt32LE(16000, 28)
    header.writeUInt16LE(2, 32)
    header.writeUInt16LE(16, 34)
    header.write('data', 36)
    header.writeUInt32LE(dataLength, 40)
    return header
  }

  const onPlaybackStatusUpdate = (s: AVPlaybackStatus) => {
    if (s.isLoaded) {
      setPosition(s.positionMillis)
      setDuration(s.durationMillis || 0)
      setIsPlaying(s.isPlaying)
      if (s.didJustFinish) {
        setIsPlaying(false)
        setPosition(0)
      }
    }
  }

  const togglePlayback = async () => {
    if (!audioUri) return
    try {
      if (soundRef.current) {
        const s = await soundRef.current.getStatusAsync()
        if (s.isLoaded) {
          isPlaying
            ? await soundRef.current.pauseAsync()
            : await soundRef.current.playAsync()
          return
        }
      }
      const { sound } = await Audio.Sound.createAsync(
        { uri: audioUri },
        { shouldPlay: true },
        onPlaybackStatusUpdate,
      )
      soundRef.current = sound
    } catch (e) {
      console.warn('Playback Error:', e)
    }
  }

  const exportRecording = async () => {
    if (!audioUri) return
    try {
      const canShare = await Sharing.isAvailableAsync()
      if (canShare)
        await Sharing.shareAsync(audioUri, {
          mimeType: 'audio/wav',
          dialogTitle: 'Export Heart Sound',
        })
    } catch (e) {
      Alert.alert('Export Error', 'Failed to share file.')
    }
  }

  return (
    <SafeAreaView style={styles.container}>
      <Modal visible={isTransferring} transparent={true} animationType='fade'>
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <ActivityIndicator size='large' color='#3498db' />
            <Text style={styles.modalTitle}>Processing Recording</Text>
            <Text style={styles.modalSubtitle}>
              Extracting Audio & BPM Sync...
            </Text>
            <View style={styles.modalProgressContainer}>
              <View
                style={[
                  styles.modalProgressBar,
                  { width: `${progress * 100}%` },
                ]}
              />
            </View>
            <Text style={styles.modalPercentage}>
              {Math.round(progress * 100)}%
            </Text>
          </View>
        </View>
      </Modal>

      <View style={styles.header}>
        <Text style={styles.title}>CardioScope</Text>
        <Text
          style={[
            styles.subtitle,
            { color: connectedDevice ? '#3498db' : '#FF3B30' },
          ]}
        >
          {status}
        </Text>
      </View>

      <ScrollView style={styles.dashboard}>
        <View style={styles.darkCard}>
          <View style={styles.cardHeaderRow}>
            <Text style={styles.cardTitle}>Stethoscope Analysis</Text>
            <Ionicons
              name='pulse'
              size={20}
              color={isPlaying ? '#3498db' : '#555'}
            />
          </View>

          {audioUri ? (
            <View>
              <Text style={styles.hrValueText}>
                {recordedBPM}{' '}
                <Text style={{ fontSize: 22, color: '#3498db' }}>bpm</Text>
              </Text>

              {/* THE SYNCED ECG - Controlled by audio playback */}
              <SyncedECG bpm={recordedBPM} active={isPlaying} />

              <Phonocardiogram audioBuffer={cleanAudioBuffer.current} />

              <View style={styles.timeRow}>
                <Text style={styles.timeText}>
                  {Math.floor(position / 1000)}s
                </Text>
                <Text style={styles.timeText}>10.0s</Text>
              </View>
              <Slider
                style={{ width: '100%', height: 40 }}
                minimumValue={0}
                maximumValue={duration}
                value={position}
                minimumTrackTintColor='#3498db'
                thumbTintColor='#3498db'
                onSlidingComplete={(v) => soundRef.current?.setPositionAsync(v)}
              />
              <TouchableOpacity onPress={togglePlayback} style={styles.playBtn}>
                <Ionicons
                  name={isPlaying ? 'pause' : 'play'}
                  size={32}
                  color='#FFF'
                />
              </TouchableOpacity>
            </View>
          ) : (
            <View style={{ paddingVertical: 40, alignItems: 'center' }}>
              <Ionicons name='mic-circle' size={64} color='#222' />
              <Text style={styles.placeholder}>
                Awaiting hardware trigger...
              </Text>
            </View>
          )}
        </View>

        <TouchableOpacity
          onPress={exportRecording}
          style={styles.primaryButton}
        >
          <Text style={styles.primaryButtonText}>Export Recording</Text>
          <Ionicons name='share-outline' size={20} color='white' />
        </TouchableOpacity>
      </ScrollView>
    </SafeAreaView>
  )
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#FFF' },
  header: { paddingHorizontal: 25, paddingTop: 20 },
  title: { fontSize: 32, fontWeight: '900' },
  subtitle: { fontSize: 13, fontWeight: 'bold', textTransform: 'uppercase' },
  dashboard: { padding: 20 },
  darkCard: {
    backgroundColor: '#000',
    borderRadius: 24,
    padding: 25,
    marginBottom: 20,
  },
  cardHeaderRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 15,
  },
  cardTitle: { color: '#888', fontWeight: '600' },
  hrValueText: { color: '#FFF', fontSize: 58, fontWeight: 'bold' },
  playBtn: {
    alignSelf: 'center',
    backgroundColor: '#3498db',
    width: 60,
    height: 60,
    borderRadius: 30,
    justifyContent: 'center',
    alignItems: 'center',
    marginTop: 10,
  },
  primaryButton: {
    backgroundColor: '#3498db',
    padding: 18,
    borderRadius: 16,
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
  },
  primaryButtonText: {
    color: '#FFF',
    fontSize: 16,
    fontWeight: 'bold',
    marginRight: 10,
  },
  placeholder: {
    color: '#555',
    textAlign: 'center',
    marginTop: 15,
    paddingHorizontal: 20,
  },
  pcgContainer: { marginBottom: 10, marginTop: 5 },
  miniLabel: {
    color: '#3498db',
    fontSize: 10,
    fontWeight: 'bold',
    textTransform: 'uppercase',
    marginBottom: 5,
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.92)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  modalContent: {
    width: '85%',
    backgroundColor: '#1C1C1E',
    padding: 35,
    borderRadius: 28,
    alignItems: 'center',
  },
  modalTitle: {
    color: '#FFF',
    fontSize: 22,
    fontWeight: 'bold',
    marginTop: 20,
  },
  modalSubtitle: {
    color: '#888',
    fontSize: 14,
    marginTop: 5,
    textAlign: 'center',
  },
  modalProgressContainer: {
    width: '100%',
    height: 6,
    backgroundColor: '#333',
    borderRadius: 3,
    marginTop: 25,
    overflow: 'hidden',
  },
  modalProgressBar: { height: '100%', backgroundColor: '#3498db' },
  modalPercentage: {
    color: '#3498db',
    fontSize: 16,
    fontWeight: 'bold',
    marginTop: 10,
  },
  timeRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 10,
  },
  timeText: { color: '#555', fontSize: 12 },
})
