import React, { useMemo, useRef, useState, useEffect } from "react";
import {
  Text,
  TouchableOpacity,
  View,
  StyleSheet,
  ScrollView,
  Animated,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { Ionicons } from "@expo/vector-icons";
import Svg, { Path, Defs, Pattern, Rect } from "react-native-svg";

const VIEW_W = 300;
const VIEW_H = 100;
const POINTS = 260; // smoother = more CPU
const FPS = 30; // cap for perf

function clamp(n: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, n));
}

function gauss(x: number, mu: number, sigma: number) {
  const z = (x - mu) / sigma;
  return Math.exp(-0.5 * z * z);
}

// Synthetic P-QRS-T beat shape; phase is 0..1 over a heartbeat
function ecgAmp(phase: number) {
  const p = 0.12 * gauss(phase, 0.16, 0.03);
  const q = -0.25 * gauss(phase, 0.285, 0.012);
  const r = 1.15 * gauss(phase, 0.305, 0.008);
  const s = -0.35 * gauss(phase, 0.33, 0.014);
  const t = 0.32 * gauss(phase, 0.56, 0.06);
  return p + q + r + s + t;
}

function mod1(x: number) {
  return ((x % 1) + 1) % 1;
}

function ECGWaveform({ bpm }: { bpm: number }) {
  const [d, setD] = useState<string>(
    `M0 ${VIEW_H / 2} L${VIEW_W} ${VIEW_H / 2}`
  );

  const scanBarX = useRef(0);
  const pointsRef = useRef<number[]>(new Array(POINTS).fill(VIEW_H / 2));

  const bpmClamped = useMemo(() => clamp(bpm, 35, 200), [bpm]);
  const beatPeriod = useMemo(() => 60 / bpmClamped, [bpmClamped]);

  useEffect(() => {
    let raf = 0;
    let last = 0;

    const sweepDuration = 3; // seconds for the bar to cross VIEW_W
    const gapWidth = 10; // points to erase ahead of the bar

    const inGap = (i: number, idx: number) => {
      // erase [idx, idx+gapWidth)
      const start = idx;
      const end = idx + gapWidth;

      if (end < POINTS) return i >= start && i < end;
      // wrap-around
      const wrapEnd = end % POINTS;
      return i >= start || i < wrapEnd;
    };

    const tick = (tms: number) => {
      if (last === 0) last = tms;

      if (tms - last < 1000 / FPS) {
        raf = requestAnimationFrame(tick);
        return;
      }

      const delta = (tms - last) / 1000;
      last = tms;

      const nowSec = tms / 1000;

      scanBarX.current =
        (scanBarX.current + (VIEW_W / sweepDuration) * delta) % VIEW_W;

      const currentIndex = Math.floor((scanBarX.current / VIEW_W) * POINTS);

      const phase = mod1(nowSec / beatPeriod);
      const amp = ecgAmp(phase);
      const y = VIEW_H / 2 - amp * (VIEW_H * 0.38);
      pointsRef.current[currentIndex] = y;

      let path = "";
      let penDown = false;

      for (let i = 0; i < POINTS; i++) {
        const erase = inGap(i, currentIndex);
        if (erase) {
          penDown = false;
          continue;
        }

        const x = (i / (POINTS - 1)) * VIEW_W;
        const py = pointsRef.current[i];

        if (!penDown) {
          path += `M${x.toFixed(1)} ${py.toFixed(1)}`;
          penDown = true;
        } else {
          path += ` L${x.toFixed(1)} ${py.toFixed(1)}`;
        }
      }

      setD(path);
      raf = requestAnimationFrame(tick);
    };

    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [beatPeriod]);

  return (
    <View>
      <Svg height={75} width="100%" viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}>
        <Defs>
          <Pattern
            id="smallGrid"
            width="10"
            height="10"
            patternUnits="userSpaceOnUse"
          >
            <Path
              d="M 10 0 L 0 0 0 10"
              fill="none"
              stroke="#151515"
              strokeWidth="1"
            />
          </Pattern>
          <Pattern
            id="bigGrid"
            width="50"
            height="50"
            patternUnits="userSpaceOnUse"
          >
            <Rect width="50" height="50" fill="url(#smallGrid)" />
            <Path
              d="M 50 0 L 0 0 0 50"
              fill="none"
              stroke="#1f1f1f"
              strokeWidth="1.2"
            />
          </Pattern>
        </Defs>

        <Rect x="0" y="0" width={VIEW_W} height={VIEW_H} fill="url(#bigGrid)" />

        <Path
          d={d}
          fill="none"
          stroke="#3498db"
          strokeWidth="3"
          strokeLinecap="round"
          strokeLinejoin="round"
        />

        {/* scan glow */}
        <Rect
          x={scanBarX.current - 2}
          y={0}
          width={4}
          height={VIEW_H}
          fill="#3498db"
          opacity={0.25}
        />
      </Svg>
    </View>
  );
}

function PulsingHeart({ bpm }: { bpm: number }) {
  const scale = useRef(new Animated.Value(1)).current;

  useEffect(() => {
    const bpmClamped = clamp(bpm, 35, 200);
    const beatMs = 60000 / bpmClamped;

    const anim = Animated.loop(
      Animated.sequence([
        Animated.timing(scale, {
          toValue: 1.12,
          duration: beatMs * 0.12,
          useNativeDriver: true,
        }),
        Animated.timing(scale, {
          toValue: 1,
          duration: beatMs * 0.88,
          useNativeDriver: true,
        }),
      ])
    );

    anim.start();
    return () => anim.stop();
  }, [bpm, scale]);

  return (
    <Animated.View style={{ transform: [{ scale }] }}>
      <Ionicons name="heart" size={48} color="#3498db" />
    </Animated.View>
  );
}

export default function CardioScopeHome() {
  const [heartRate, setHeartRate] = useState<number>(72);

  useEffect(() => {
    const interval = setInterval(() => {
      setHeartRate((prev) => {
        const next = prev + (Math.random() > 0.5 ? 1 : -1);
        return clamp(next, 100, 110);
      });
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  return (
    <SafeAreaView style={styles.container}>
      {/* HEADER */}
      <View style={styles.header}>
        <Text style={styles.title}>CardioScope</Text>
        <Text style={styles.subtitle}>Live Monitoring Active</Text>
      </View>

      <ScrollView style={styles.dashboard}>
        {/* HEART RATE CARD */}
        <View style={styles.darkCard}>
          <View style={styles.cardHeaderRow}>
            <Text style={styles.cardTitle}>Heart Rate</Text>
            <Ionicons name="pulse" size={20} color="#3498db" />
          </View>

          <View style={styles.hrContent}>
            <PulsingHeart bpm={heartRate} />
            <Text style={styles.hrValueText}>
              {heartRate} <Text style={styles.unitText}>bpm</Text>
            </Text>
          </View>

          {/* ECG */}
          <View style={styles.waveformWrapper}>
            <ECGWaveform bpm={heartRate} />
          </View>
        </View>

        {/* DIAGNOSIS CARD */}
        <View style={styles.darkCard}>
          <View style={styles.cardHeaderRow}>
            <Text style={styles.cardTitle}>Diagnosis</Text>
            <Ionicons name="information-circle" size={20} color="#3498db" />
          </View>

          <View style={styles.diagContent}>
            <View style={styles.donutPlaceholder}>
              <Svg height="80" width="80" viewBox="0 0 100 100">
                {/* Background Circle */}
                <Path
                  d="M50,10 A40,40 0 1,1 49.9,10"
                  fill="none"
                  stroke="#1c1c1e"
                  strokeWidth="10"
                />
                {/* Progress Blue */}
                <Path
                  d="M50,10 A40,40 0 0,1 90,50"
                  fill="none"
                  stroke="#3498db"
                  strokeWidth="10"
                  strokeLinecap="round"
                />
              </Svg>
            </View>

            <View style={styles.diagLegend}>
              <View style={styles.legendRow}>
                <View style={[styles.dot, { backgroundColor: "#3498db" }]} />
                <Text style={styles.legendText}>Normal</Text>
                <Text style={styles.percentText}>94%</Text>
              </View>

              <View style={styles.legendRow}>
                <View style={[styles.dot, { backgroundColor: "#1c1c1e" }]} />
                <Text style={styles.legendText}>Bradycardia</Text>
                <Text style={styles.percentText}>4%</Text>
              </View>

              <View style={styles.legendRow}>
                <View style={[styles.dot, { backgroundColor: "#444" }]} />
                <Text style={styles.legendText}>Arrhythmia</Text>
                <Text style={styles.percentText}>2%</Text>
              </View>
            </View>
          </View>
        </View>

        {/* ACTIONS */}
        <TouchableOpacity style={styles.primaryButton}>
          <Text style={styles.primaryButtonText}>Export Recording</Text>
          <Ionicons name="share-outline" size={20} color="white" />
        </TouchableOpacity>

        <TouchableOpacity style={styles.secondaryButton}>
          <Text style={styles.secondaryButtonText}>View History</Text>
          <Ionicons name="time-outline" size={20} color="#3498db" />
        </TouchableOpacity>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#FFFFFF",
  },
  header: {
    paddingHorizontal: 25,
    paddingTop: 20,
    paddingBottom: 15,
  },
  title: {
    fontSize: 34,
    fontWeight: "900",
    color: "#000",
    letterSpacing: -1,
  },
  subtitle: {
    fontSize: 14,
    color: "#3498db",
    fontWeight: "700",
    textTransform: "uppercase",
    letterSpacing: 1,
  },
  dashboard: {
    flex: 1,
    padding: 20,
  },
  darkCard: {
    backgroundColor: "#000000",
    borderRadius: 20,
    padding: 24,
    marginBottom: 20,
    shadowColor: "#3498db",
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.1,
    shadowRadius: 20,
  },
  cardHeaderRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 15,
  },
  cardTitle: {
    color: "#FFF",
    fontSize: 18,
    fontWeight: "600",
  },
  hrContent: {
    flexDirection: "row",
    alignItems: "flex-end",
    gap: 10,
  },
  hrValueText: {
    color: "#FFF",
    fontSize: 54,
    fontWeight: "800",
    lineHeight: 54,
  },
  unitText: {
    fontSize: 18,
    color: "#3498db",
    fontWeight: "600",
  },
  waveformWrapper: {
    marginTop: 15,
  },
  diagContent: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  donutPlaceholder: {
    width: 80,
    height: 80,
    justifyContent: "center",
    alignItems: "center",
  },
  diagLegend: {
    flex: 1,
    paddingLeft: 20,
    gap: 10,
  },
  legendRow: {
    flexDirection: "row",
    alignItems: "center",
  },
  dot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: 10,
  },
  legendText: {
    color: "#AAA",
    fontSize: 14,
    fontWeight: "500",
    flex: 1,
  },
  percentText: {
    color: "#FFF",
    fontWeight: "700",
    fontSize: 14,
  },
  primaryButton: {
    backgroundColor: "#3498db",
    padding: 20,
    borderRadius: 16,
    flexDirection: "row",
    justifyContent: "center",
    alignItems: "center",
    marginBottom: 12,
  },
  primaryButtonText: {
    color: "#FFF",
    fontSize: 18,
    fontWeight: "800",
    marginRight: 10,
  },
  secondaryButton: {
    backgroundColor: "#000",
    padding: 20,
    borderRadius: 16,
    flexDirection: "row",
    justifyContent: "center",
    alignItems: "center",
    borderWidth: 1,
    borderColor: "#333",
  },
  secondaryButtonText: {
    color: "#3498db",
    fontSize: 18,
    fontWeight: "700",
    marginRight: 10,
  },
});
