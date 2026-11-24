// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.2 (64-bit)
// Tool Version Limit: 2019.12
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// ==============================================================
#ifndef __linux__

#include "xstatus.h"
#include "xparameters.h"
#include "xradar_cnn_stream_full.h"

extern XRadar_cnn_stream_full_Config XRadar_cnn_stream_full_ConfigTable[];

XRadar_cnn_stream_full_Config *XRadar_cnn_stream_full_LookupConfig(u16 DeviceId) {
	XRadar_cnn_stream_full_Config *ConfigPtr = NULL;

	int Index;

	for (Index = 0; Index < XPAR_XRADAR_CNN_STREAM_FULL_NUM_INSTANCES; Index++) {
		if (XRadar_cnn_stream_full_ConfigTable[Index].DeviceId == DeviceId) {
			ConfigPtr = &XRadar_cnn_stream_full_ConfigTable[Index];
			break;
		}
	}

	return ConfigPtr;
}

int XRadar_cnn_stream_full_Initialize(XRadar_cnn_stream_full *InstancePtr, u16 DeviceId) {
	XRadar_cnn_stream_full_Config *ConfigPtr;

	Xil_AssertNonvoid(InstancePtr != NULL);

	ConfigPtr = XRadar_cnn_stream_full_LookupConfig(DeviceId);
	if (ConfigPtr == NULL) {
		InstancePtr->IsReady = 0;
		return (XST_DEVICE_NOT_FOUND);
	}

	return XRadar_cnn_stream_full_CfgInitialize(InstancePtr, ConfigPtr);
}

#endif

