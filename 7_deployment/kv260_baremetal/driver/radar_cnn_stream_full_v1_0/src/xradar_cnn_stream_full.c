// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.2 (64-bit)
// Tool Version Limit: 2019.12
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// ==============================================================
/***************************** Include Files *********************************/
#include "xradar_cnn_stream_full.h"

/************************** Function Implementation *************************/
#ifndef __linux__
int XRadar_cnn_stream_full_CfgInitialize(XRadar_cnn_stream_full *InstancePtr, XRadar_cnn_stream_full_Config *ConfigPtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(ConfigPtr != NULL);

    InstancePtr->Control_BaseAddress = ConfigPtr->Control_BaseAddress;
    InstancePtr->IsReady = XIL_COMPONENT_IS_READY;

    return XST_SUCCESS;
}
#endif

void XRadar_cnn_stream_full_Start(XRadar_cnn_stream_full *InstancePtr) {
    u32 Data;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XRadar_cnn_stream_full_ReadReg(InstancePtr->Control_BaseAddress, XRADAR_CNN_STREAM_FULL_CONTROL_ADDR_AP_CTRL) & 0x80;
    XRadar_cnn_stream_full_WriteReg(InstancePtr->Control_BaseAddress, XRADAR_CNN_STREAM_FULL_CONTROL_ADDR_AP_CTRL, Data | 0x01);
}

u32 XRadar_cnn_stream_full_IsDone(XRadar_cnn_stream_full *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XRadar_cnn_stream_full_ReadReg(InstancePtr->Control_BaseAddress, XRADAR_CNN_STREAM_FULL_CONTROL_ADDR_AP_CTRL);
    return (Data >> 1) & 0x1;
}

u32 XRadar_cnn_stream_full_IsIdle(XRadar_cnn_stream_full *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XRadar_cnn_stream_full_ReadReg(InstancePtr->Control_BaseAddress, XRADAR_CNN_STREAM_FULL_CONTROL_ADDR_AP_CTRL);
    return (Data >> 2) & 0x1;
}

u32 XRadar_cnn_stream_full_IsReady(XRadar_cnn_stream_full *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XRadar_cnn_stream_full_ReadReg(InstancePtr->Control_BaseAddress, XRADAR_CNN_STREAM_FULL_CONTROL_ADDR_AP_CTRL);
    // check ap_start to see if the pcore is ready for next input
    return !(Data & 0x1);
}

void XRadar_cnn_stream_full_EnableAutoRestart(XRadar_cnn_stream_full *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XRadar_cnn_stream_full_WriteReg(InstancePtr->Control_BaseAddress, XRADAR_CNN_STREAM_FULL_CONTROL_ADDR_AP_CTRL, 0x80);
}

void XRadar_cnn_stream_full_DisableAutoRestart(XRadar_cnn_stream_full *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XRadar_cnn_stream_full_WriteReg(InstancePtr->Control_BaseAddress, XRADAR_CNN_STREAM_FULL_CONTROL_ADDR_AP_CTRL, 0);
}

void XRadar_cnn_stream_full_Set_input_r(XRadar_cnn_stream_full *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XRadar_cnn_stream_full_WriteReg(InstancePtr->Control_BaseAddress, XRADAR_CNN_STREAM_FULL_CONTROL_ADDR_INPUT_R_DATA, (u32)(Data));
    XRadar_cnn_stream_full_WriteReg(InstancePtr->Control_BaseAddress, XRADAR_CNN_STREAM_FULL_CONTROL_ADDR_INPUT_R_DATA + 4, (u32)(Data >> 32));
}

u64 XRadar_cnn_stream_full_Get_input_r(XRadar_cnn_stream_full *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XRadar_cnn_stream_full_ReadReg(InstancePtr->Control_BaseAddress, XRADAR_CNN_STREAM_FULL_CONTROL_ADDR_INPUT_R_DATA);
    Data += (u64)XRadar_cnn_stream_full_ReadReg(InstancePtr->Control_BaseAddress, XRADAR_CNN_STREAM_FULL_CONTROL_ADDR_INPUT_R_DATA + 4) << 32;
    return Data;
}

void XRadar_cnn_stream_full_Set_logits(XRadar_cnn_stream_full *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XRadar_cnn_stream_full_WriteReg(InstancePtr->Control_BaseAddress, XRADAR_CNN_STREAM_FULL_CONTROL_ADDR_LOGITS_DATA, (u32)(Data));
    XRadar_cnn_stream_full_WriteReg(InstancePtr->Control_BaseAddress, XRADAR_CNN_STREAM_FULL_CONTROL_ADDR_LOGITS_DATA + 4, (u32)(Data >> 32));
}

u64 XRadar_cnn_stream_full_Get_logits(XRadar_cnn_stream_full *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XRadar_cnn_stream_full_ReadReg(InstancePtr->Control_BaseAddress, XRADAR_CNN_STREAM_FULL_CONTROL_ADDR_LOGITS_DATA);
    Data += (u64)XRadar_cnn_stream_full_ReadReg(InstancePtr->Control_BaseAddress, XRADAR_CNN_STREAM_FULL_CONTROL_ADDR_LOGITS_DATA + 4) << 32;
    return Data;
}

void XRadar_cnn_stream_full_InterruptGlobalEnable(XRadar_cnn_stream_full *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XRadar_cnn_stream_full_WriteReg(InstancePtr->Control_BaseAddress, XRADAR_CNN_STREAM_FULL_CONTROL_ADDR_GIE, 1);
}

void XRadar_cnn_stream_full_InterruptGlobalDisable(XRadar_cnn_stream_full *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XRadar_cnn_stream_full_WriteReg(InstancePtr->Control_BaseAddress, XRADAR_CNN_STREAM_FULL_CONTROL_ADDR_GIE, 0);
}

void XRadar_cnn_stream_full_InterruptEnable(XRadar_cnn_stream_full *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XRadar_cnn_stream_full_ReadReg(InstancePtr->Control_BaseAddress, XRADAR_CNN_STREAM_FULL_CONTROL_ADDR_IER);
    XRadar_cnn_stream_full_WriteReg(InstancePtr->Control_BaseAddress, XRADAR_CNN_STREAM_FULL_CONTROL_ADDR_IER, Register | Mask);
}

void XRadar_cnn_stream_full_InterruptDisable(XRadar_cnn_stream_full *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XRadar_cnn_stream_full_ReadReg(InstancePtr->Control_BaseAddress, XRADAR_CNN_STREAM_FULL_CONTROL_ADDR_IER);
    XRadar_cnn_stream_full_WriteReg(InstancePtr->Control_BaseAddress, XRADAR_CNN_STREAM_FULL_CONTROL_ADDR_IER, Register & (~Mask));
}

void XRadar_cnn_stream_full_InterruptClear(XRadar_cnn_stream_full *InstancePtr, u32 Mask) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XRadar_cnn_stream_full_WriteReg(InstancePtr->Control_BaseAddress, XRADAR_CNN_STREAM_FULL_CONTROL_ADDR_ISR, Mask);
}

u32 XRadar_cnn_stream_full_InterruptGetEnabled(XRadar_cnn_stream_full *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XRadar_cnn_stream_full_ReadReg(InstancePtr->Control_BaseAddress, XRADAR_CNN_STREAM_FULL_CONTROL_ADDR_IER);
}

u32 XRadar_cnn_stream_full_InterruptGetStatus(XRadar_cnn_stream_full *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XRadar_cnn_stream_full_ReadReg(InstancePtr->Control_BaseAddress, XRADAR_CNN_STREAM_FULL_CONTROL_ADDR_ISR);
}

