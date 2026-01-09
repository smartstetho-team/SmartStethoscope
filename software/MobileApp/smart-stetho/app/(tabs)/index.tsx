import React, { useEffect, useRef, useState } from "react";
import {
  Alert,
  FlatList,
  Platform,
  Text,
  TouchableOpacity,
  View,
  StyleSheet,
  ActivityIndicator,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";

// Type-only import (safe for web)
import type { Device, Subscription } from "react-native-ble-plx";

const isWeb = Platform.OS === "web";

/**
 * LAZY LOAD BLE STRATEGY
 */
let BlePlx: any = null;
function loadBlePlx() {
  if (isWeb) return null;
  if (BlePlx) return BlePlx;
  try {
    BlePlx = require("react-native-ble-plx");
    return BlePlx;
  } catch (e) {
    return null;
  }
}

let managerSingleton: any = null;
function getManager(): any | null {
  if (isWeb) return null;
  if (managerSingleton) return managerSingleton;
  const mod = loadBlePlx();
  if (!mod?.BleManager) return null;
  try {
    managerSingleton = new mod.BleManager();
    return managerSingleton;
  } catch (e) {
    return null;
  }
}

export default function App() {
  const [bleState, setBleState] = useState<string>(isWeb ? "Web" : "Unknown");
  const [scanning, setScanning] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [connectingId, setConnectingId] = useState<string | null>(null);
  const [devices, setDevices] = useState<Device[]>([]);
  const [connected, setConnected] = useState<Device | null>(null);

  const scanTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);
  const stateSub = useRef<Subscription | null>(null);
  const disconnectSub = useRef<Subscription | null>(null);

  useEffect(() => {
    if (isWeb) return;
    const manager = getManager();
    if (!manager) {
      setBleState("BLE Unavailable");
      return;
    }

    // Monitor Bluetooth hardware state
    stateSub.current = manager.onStateChange((s: any) => {
      setBleState(String(s));
    }, true);

    return () => {
      stateSub.current?.remove?.();
      disconnectSub.current?.remove?.();
      stopScan();
    };
  }, []);

  const stopScan = () => {
    const manager = getManager();
    manager?.stopDeviceScan?.();
    setScanning(false);
    if (scanTimeout.current) {
      clearTimeout(scanTimeout.current);
      scanTimeout.current = null;
    }
  };

  const startScan = async () => {
    const manager = getManager();
    if (!manager || bleState !== "PoweredOn") {
      Alert.alert(
        "Error",
        "Ensure Bluetooth is ON and you're on a physical device."
      );
      return;
    }
    setDevices([]);
    setScanning(true);
    manager.startDeviceScan(
      null,
      { allowDuplicates: false },
      (error: any, device: Device | null) => {
        if (error) {
          stopScan();
          return;
        }
        if (device) {
          setDevices((prev) => {
            const exists = prev.find((d) => d.id === device.id);
            return exists ? prev : [device, ...prev];
          });
        }
      }
    );
    scanTimeout.current = setTimeout(stopScan, 10000);
  };

  const cancelConnectionAttempt = async (deviceId: string) => {
    const manager = getManager();
    try {
      await manager.cancelDeviceConnection(deviceId);
      setIsConnecting(false);
      setConnectingId(null);
      Alert.alert("Cancelled", "Connection attempt stopped.");
    } catch (e) {
      setIsConnecting(false);
      setConnectingId(null);
    }
  };

  const connectTo = async (device: Device) => {
    if (isConnecting) return;
    const manager = getManager();
    setIsConnecting(true);
    setConnectingId(device.id);

    // Safety Timeout: 15 seconds
    const timer = setTimeout(() => {
      if (isConnecting) {
        cancelConnectionAttempt(device.id);
        Alert.alert("Timeout", "Device did not respond in time.");
      }
    }, 15000);

    try {
      stopScan();
      await new Promise((r) => setTimeout(r, 400));

      const connectedDevice = await manager.connectToDevice(device.id);
      await connectedDevice.discoverAllServicesAndCharacteristics();

      clearTimeout(timer);

      // Listen for accidental disconnections (out of range, battery dead)
      disconnectSub.current = manager.onDeviceDisconnected(
        connectedDevice.id,
        () => {
          setConnected(null);
          Alert.alert("Lost Connection", "The device was disconnected.");
        }
      );

      setConnected(connectedDevice);
    } catch (e: any) {
      clearTimeout(timer);
      if (e.message !== "Operation was cancelled") {
        Alert.alert("Connection Failed", e.message);
      }
    } finally {
      setIsConnecting(false);
      setConnectingId(null);
    }
  };

  const logInternalDetails = async () => {
    if (!connected) return;
    try {
      const services = await connected.services();
      let report = `Device: ${connected.name || connected.id}\n\n`;
      for (const service of services) {
        const sUUID = service.uuid.split("-")[0].toUpperCase();
        report += `Svc: ${sUUID}\n`;
        const chars = await service.characteristics();
        chars.forEach((c) => {
          const cUUID = c.uuid.split("-")[0].toUpperCase();
          report += `  └─ Char: ${cUUID} (Read: ${c.isReadable ? "Y" : "N"})\n`;
        });
        report += `\n`;
      }
      Alert.alert("GATT Service Map", report);
      console.log(report);
    } catch (e: any) {
      Alert.alert("Discovery Error", e.message);
    }
  };

  const disconnect = async () => {
    if (connected) {
      await getManager()?.cancelDeviceConnection(connected.id);
      disconnectSub.current?.remove?.();
      setConnected(null);
      Alert.alert("Disconnected", "Hardware released.");
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      {/* HEADER */}
      <View style={styles.header}>
        <Text style={styles.title}>SmartStetho BLE</Text>
        <Text style={styles.subtitle}>Bluetooth: {bleState}</Text>
        {connected && (
          <Text style={styles.connectedStatus}>
            🟢 Connected: {connected.name || connected.id}
          </Text>
        )}
      </View>

      {/* ACTIONS */}
      <View style={styles.actionContainer}>
        {!connected ? (
          <TouchableOpacity
            style={[styles.button, scanning && styles.buttonDisabled]}
            onPress={startScan}
            disabled={scanning}
          >
            {scanning ? (
              <ActivityIndicator color="#fff" />
            ) : (
              <Text style={styles.buttonText}>Scan for Stethoscope</Text>
            )}
          </TouchableOpacity>
        ) : (
          <View style={styles.connectedActions}>
            <TouchableOpacity
              style={[styles.button, { backgroundColor: "#5856D6" }]}
              onPress={logInternalDetails}
            >
              <Text style={styles.buttonText}>Explore Services</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.button, styles.dangerButton]}
              onPress={disconnect}
            >
              <Text style={styles.buttonText}>Disconnect Device</Text>
            </TouchableOpacity>
          </View>
        )}
      </View>

      {/* CONNECTING OVERLAY */}
      {isConnecting && (
        <View style={styles.loadingOverlay}>
          <View style={styles.loadingCard}>
            <ActivityIndicator size="large" color="#007AFF" />
            <Text style={styles.loadingText}>Establishing Bridge...</Text>
            <TouchableOpacity
              style={styles.cancelBtn}
              onPress={() =>
                connectingId && cancelConnectionAttempt(connectingId)
              }
            >
              <Text style={styles.cancelBtnText}>Cancel Attempt</Text>
            </TouchableOpacity>
          </View>
        </View>
      )}

      {/* LIST */}
      <View style={{ flex: 1 }}>
        <Text style={styles.sectionLabel}>
          Nearby Devices ({devices.length})
        </Text>
        <FlatList
          data={devices}
          keyExtractor={(item) => item.id}
          contentContainerStyle={styles.list}
          renderItem={({ item }) => (
            <TouchableOpacity
              style={styles.deviceCard}
              onPress={() => connectTo(item)}
              disabled={isConnecting}
            >
              <View style={{ flex: 1 }}>
                <Text style={styles.deviceName}>
                  {item.name || "Unnamed Device"}
                </Text>
                <Text style={styles.deviceId}>{item.id}</Text>
              </View>
              <View style={styles.rssiBadge}>
                <Text style={styles.deviceRssi}>{item.rssi || "--"} dBm</Text>
              </View>
            </TouchableOpacity>
          )}
          ListEmptyComponent={
            !scanning && (
              <Text style={styles.emptyText}>
                No devices found. Tap Scan to search.
              </Text>
            )
          }
        />
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#f2f2f7" },
  header: {
    padding: 20,
    backgroundColor: "#fff",
    borderBottomWidth: 1,
    borderBottomColor: "#e5e5ea",
  },
  title: { fontSize: 24, fontWeight: "800", color: "#000" },
  subtitle: { fontSize: 14, color: "#8e8e93", marginTop: 4 },
  connectedStatus: { marginTop: 10, color: "#34C759", fontWeight: "600" },
  actionContainer: {
    padding: 15,
    backgroundColor: "#fff",
    borderBottomWidth: 1,
    borderBottomColor: "#e5e5ea",
  },
  connectedActions: { gap: 10 },
  button: {
    backgroundColor: "#007AFF",
    padding: 16,
    borderRadius: 12,
    alignItems: "center",
    justifyContent: "center",
    height: 56,
  },
  buttonDisabled: { backgroundColor: "#c7c7cc" },
  dangerButton: { backgroundColor: "#FF3B30" },
  buttonText: { color: "#fff", fontSize: 16, fontWeight: "600" },
  sectionLabel: {
    fontSize: 13,
    textTransform: "uppercase",
    color: "#8e8e93",
    marginHorizontal: 15,
    marginTop: 20,
    marginBottom: 8,
  },
  list: { paddingHorizontal: 15, paddingBottom: 40 },
  deviceCard: {
    backgroundColor: "#fff",
    padding: 16,
    borderRadius: 14,
    marginBottom: 12,
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    elevation: 1,
  },
  deviceName: { fontSize: 17, fontWeight: "600", color: "#1c1c1e" },
  deviceId: { fontSize: 13, color: "#8e8e93", marginTop: 2 },
  rssiBadge: {
    backgroundColor: "#f2f2f7",
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 6,
  },
  deviceRssi: { fontSize: 13, fontWeight: "600", color: "#007AFF" },
  emptyText: { textAlign: "center", marginTop: 40, color: "#8e8e93" },

  // OVERLAY STYLES
  loadingOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: "rgba(0,0,0,0.4)",
    justifyContent: "center",
    alignItems: "center",
    zIndex: 1000,
  },
  loadingCard: {
    backgroundColor: "#fff",
    padding: 30,
    borderRadius: 20,
    alignItems: "center",
    width: "80%",
  },
  loadingText: {
    marginTop: 15,
    fontSize: 16,
    fontWeight: "600",
    color: "#1c1c1e",
  },
  cancelBtn: {
    marginTop: 20,
    paddingVertical: 10,
    paddingHorizontal: 20,
    backgroundColor: "#ff3b30",
    borderRadius: 8,
  },
  cancelBtnText: { color: "#fff", fontWeight: "bold" },
});
