import React, { useState } from "react";
import {
  View,
  StyleSheet,
  ScrollView,
  Switch,
  TouchableOpacity,
  Alert,
  Modal,
} from "react-native";
import { ThemedText } from "@/components/themed-text";
import { ThemedView } from "@/components/themed-view";
import { IconSymbol } from "@/components/ui/icon-symbol";
import { Fonts } from "@/constants/theme";

// Import the scanner component we built
import BleScannerModal from "@/components/BleScannerModal";
import { getManager } from "@/utils/ble-manager";

export default function SettingsScreen() {
  const [isCloudSync, setIsCloudSync] = useState(true);
  const [isNoiseCancelling, setIsNoiseCancelling] = useState(false);

  // Bluetooth States
  const [isScannerVisible, setIsScannerVisible] = useState(false);
  const [connectedDevice, setConnectedDevice] = useState<any>(null);

  const Section = ({
    title,
    children,
  }: {
    title: string;
    children: React.ReactNode;
  }) => (
    <View style={styles.section}>
      <ThemedText style={styles.sectionHeader}>{title}</ThemedText>
      <View style={styles.card}>{children}</View>
    </View>
  );

  const SettingRow = ({
    icon,
    label,
    value,
    isLast = false,
    onPress,
    toggle,
    valueColor,
  }: any) => (
    <TouchableOpacity
      style={[styles.row, isLast && { borderBottomWidth: 0 }]}
      onPress={onPress}
      disabled={!onPress && toggle === undefined}
    >
      <View style={styles.rowLeft}>
        <View style={styles.iconContainer}>
          <IconSymbol name={icon} size={20} color="#007AFF" />
        </View>
        <ThemedText style={styles.rowLabel}>{label}</ThemedText>
      </View>
      <View style={styles.rowRight}>
        {toggle !== undefined ? (
          <Switch
            value={toggle}
            onValueChange={onPress}
            trackColor={{ true: "#34C759" }}
          />
        ) : (
          <>
            <ThemedText
              style={[styles.rowValue, valueColor && { color: valueColor }]}
            >
              {value}
            </ThemedText>
            <IconSymbol name="chevron.right" size={14} color="#C7C7CC" />
          </>
        )}
      </View>
    </TouchableOpacity>
  );

  const handleUnpairAndReset = () => {
    Alert.alert(
      "Unpair & Reset",
      "This will disconnect the stethoscope and wipe local recordings. Are you sure?",
      [
        { text: "Cancel", style: "cancel" },
        {
          text: "Unpair",
          style: "destructive",
          onPress: async () => {
            if (connectedDevice) {
              try {
                const manager = getManager();
                // 1. Physically break the BLE connection
                await manager.cancelDeviceConnection(connectedDevice.id);

                // 2. Clear the local app state
                setConnectedDevice(null);

                console.log("Device unpaired successfully");
              } catch (e) {
                console.error("Failed to unpair:", e);
                // Fallback: Clear state anyway if device is already gone
                setConnectedDevice(null);
              }
            } else {
              Alert.alert(
                "No Device",
                "There is no stethoscope currently paired."
              );
            }
          },
        },
      ]
    );
  };

  // Need to read battery service: Battery Service (0x180F)
  const handleBatteryLife = () => {
    console.log("90%");
  };

  // Add this at the top of explore.tsx
  const decodeBleString = (base64Value: string | null) => {
    if (!base64Value) return "No Data";
    try {
      return atob(base64Value); // Converts Base64 to plain text
    } catch (e) {
      return "Decode Error";
    }
  };

  const readManufacturer = async () => {
    if (!connectedDevice) {
      Alert.alert("Not Connected", "Please connect to your MacBook first.");
      return;
    }

    try {
      // 1. Target the Device Info Service (180A) and Manufacturer Char (2A29)
      const characteristic = await connectedDevice.readCharacteristicForService(
        "0000180A-0000-1000-8000-00805f9b34fb",
        "00002A29-0000-1000-8000-00805f9b34fb"
      );

      // 2. Decode the result
      const name = decodeBleString(characteristic.value);

      // 3. Show the result
      Alert.alert("MacBook Info", `Manufacturer: ${name}`);
    } catch (e: any) {
      console.log("Read failed:", e);
      Alert.alert(
        "Read Failed",
        "Make sure the device is still in range and connected."
      );
    }
  };

  return (
    <ThemedView style={styles.container}>
      <View style={styles.header}>
        <ThemedText type="title" style={styles.headerTitle}>
          Settings
        </ThemedText>
      </View>

      <ScrollView contentContainerStyle={styles.scrollContent}>
        <Section title="DEVICE STATUS">
          <SettingRow
            icon="stethoscope"
            label="Hardware Link"
            // Displays device name or "Not Connected"
            value={
              connectedDevice
                ? connectedDevice.name || "Connected"
                : "Not Connected"
            }
            valueColor={connectedDevice ? "#34C759" : "#8E8E93"}
            onPress={() => setIsScannerVisible(true)}
          />
          <SettingRow
            icon="battery.100"
            label="Stetho Battery"
            value={connectedDevice ? "84%" : "--"}
            isLast={true}
            onPress={() => {
              if (connectedDevice) {
                readManufacturer(); // Read from Mac if already linked
              } else {
                setIsScannerVisible(true); // Open scanner if not linked
              }
            }}
          />
        </Section>

        <Section title="SIGNAL PROCESSING">
          <SettingRow
            icon="waveform.path"
            label="Active Noise Control"
            toggle={isNoiseCancelling}
            onPress={() => setIsNoiseCancelling(!isNoiseCancelling)}
          />
          <SettingRow
            icon="tuningfork"
            label="Acoustic Mode"
            value="Diaphragm"
            isLast={true}
          />
        </Section>

        <Section title="DATA MANAGEMENT">
          <SettingRow
            icon="icloud.fill"
            label="Cloud Backup"
            toggle={isCloudSync}
            onPress={() => setIsCloudSync(!isCloudSync)}
          />
          <SettingRow
            icon="lock.shield.fill"
            label="Encryption"
            value="AES-256"
            isLast={true}
          />
        </Section>

        <TouchableOpacity
          style={styles.dangerZone}
          onPress={handleUnpairAndReset} // Updated this
        >
          <ThemedText style={styles.dangerText}>
            Unpair & Reset Device
          </ThemedText>
        </TouchableOpacity>

        <ThemedText style={styles.versionLabel}>
          SmartStetho Platform • Version 1.0.2
        </ThemedText>
      </ScrollView>

      {/* BLUETOOTH SCANNER MODAL */}
      <Modal
        animationType="slide"
        presentationStyle="pageSheet"
        visible={isScannerVisible}
        onRequestClose={() => setIsScannerVisible(false)}
      >
        <BleScannerModal
          onDeviceConnected={(device: any) => {
            setConnectedDevice(device);
            setIsScannerVisible(false);
          }}
          onClose={() => setIsScannerVisible(false)}
        />
      </Modal>
    </ThemedView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#F2F2F7" },
  header: {
    paddingTop: 60,
    paddingHorizontal: 20,
    paddingBottom: 20,
    backgroundColor: "#FFF",
  },
  headerTitle: {
    fontFamily: Fonts.rounded,
    fontSize: 34,
    fontWeight: "bold",
  },
  scrollContent: { paddingBottom: 40 },
  section: { marginTop: 24, paddingHorizontal: 16 },
  sectionHeader: {
    fontSize: 13,
    color: "#8E8E93",
    marginBottom: 8,
    marginLeft: 8,
    fontWeight: "600",
  },
  card: {
    backgroundColor: "#FFF",
    borderRadius: 12,
    overflow: "hidden",
  },
  row: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: "#C6C6C8",
  },
  rowLeft: { flexDirection: "row", alignItems: "center" },
  iconContainer: {
    width: 32,
    height: 32,
    borderRadius: 6,
    backgroundColor: "#E1EFFF",
    justifyContent: "center",
    alignItems: "center",
    marginRight: 12,
  },
  rowLabel: { fontSize: 17, color: "#000" },
  rowRight: { flexDirection: "row", alignItems: "center", gap: 8 },
  rowValue: { fontSize: 17, color: "#8E8E93" },
  dangerZone: {
    marginTop: 32,
    marginHorizontal: 16,
    backgroundColor: "#FFF",
    borderRadius: 12,
    padding: 16,
    alignItems: "center",
  },
  dangerText: { color: "#FF3B30", fontSize: 17, fontWeight: "600" },
  versionLabel: {
    textAlign: "center",
    marginTop: 20,
    color: "#8E8E93",
    fontSize: 13,
  },
});
