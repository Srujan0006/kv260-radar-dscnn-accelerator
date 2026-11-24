// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.2 (64-bit)
// Tool Version Limit: 2019.12
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// ==============================================================
#ifndef XRADAR_CNN_STREAM_FULL_H
#define XRADAR_CNN_STREAM_FULL_H

#ifdef __cplusplus
extern "C" {
#endif

/***************************** Include Files *********************************/
#ifndef __linux__
#include "xil_types.h"
#include "xil_assert.h"
#include "xstatus.h"
#include "xil_io.h"
#else
#include <stdint.h>
#include <assert.h>
#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stddef.h>
#endif
#include "xradar_cnn_stream_full_hw.h"

/**************************** Type Definitions ******************************/
#ifdef __linux__
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
#else
typedef struct {
    u16 DeviceId;
    u64 Control_BaseAddress;
} XRadar_cnn_stream_full_Config;
#endif

typedef struct {
    u64 Control_BaseAddress;
    u32 IsReady;
} XRadar_cnn_stream_full;

typedef u32 word_type;

/***************** Macros (Inline Functions) Definitions *********************/
#ifndef __linux__
#define XRadar_cnn_stream_full_WriteReg(BaseAddress, RegOffset, Data) \
    Xil_Out32((BaseAddress) + (RegOffset), (u32)(Data))
#define XRadar_cnn_stream_full_ReadReg(BaseAddress, RegOffset) \
    Xil_In32((BaseAddress) + (RegOffset))
#else
#define XRadar_cnn_stream_full_WriteReg(BaseAddress, RegOffset, Data) \
    *(volatile u32*)((BaseAddress) + (RegOffset)) = (u32)(Data)
#define XRadar_cnn_stream_full_ReadReg(BaseAddress, RegOffset) \
    *(volatile u32*)((BaseAddress) + (RegOffset))

#define Xil_AssertVoid(expr)    assert(expr)
#define Xil_AssertNonvoid(expr) assert(expr)

#define XST_SUCCESS             0
#define XST_DEVICE_NOT_FOUND    2
#define XST_OPEN_DEVICE_FAILED  3
#define XIL_COMPONENT_IS_READY  1
#endif

/************************** Function Prototypes *****************************/
#ifndef __linux__
int XRadar_cnn_stream_full_Initialize(XRadar_cnn_stream_full *InstancePtr, u16 DeviceId);
XRadar_cnn_stream_full_Config* XRadar_cnn_stream_full_LookupConfig(u16 DeviceId);
int XRadar_cnn_stream_full_CfgInitialize(XRadar_cnn_stream_full *InstancePtr, XRadar_cnn_stream_full_Config *ConfigPtr);
#else
int XRadar_cnn_stream_full_Initialize(XRadar_cnn_stream_full *InstancePtr, const char* InstanceName);
int XRadar_cnn_stream_full_Release(XRadar_cnn_stream_full *InstancePtr);
#endif

void XRadar_cnn_stream_full_Start(XRadar_cnn_stream_full *InstancePtr);
u32 XRadar_cnn_stream_full_IsDone(XRadar_cnn_stream_full *InstancePtr);
u32 XRadar_cnn_stream_full_IsIdle(XRadar_cnn_stream_full *InstancePtr);
u32 XRadar_cnn_stream_full_IsReady(XRadar_cnn_stream_full *InstancePtr);
void XRadar_cnn_stream_full_EnableAutoRestart(XRadar_cnn_stream_full *InstancePtr);
void XRadar_cnn_stream_full_DisableAutoRestart(XRadar_cnn_stream_full *InstancePtr);

void XRadar_cnn_stream_full_Set_input_r(XRadar_cnn_stream_full *InstancePtr, u64 Data);
u64 XRadar_cnn_stream_full_Get_input_r(XRadar_cnn_stream_full *InstancePtr);
void XRadar_cnn_stream_full_Set_logits(XRadar_cnn_stream_full *InstancePtr, u64 Data);
u64 XRadar_cnn_stream_full_Get_logits(XRadar_cnn_stream_full *InstancePtr);

void XRadar_cnn_stream_full_InterruptGlobalEnable(XRadar_cnn_stream_full *InstancePtr);
void XRadar_cnn_stream_full_InterruptGlobalDisable(XRadar_cnn_stream_full *InstancePtr);
void XRadar_cnn_stream_full_InterruptEnable(XRadar_cnn_stream_full *InstancePtr, u32 Mask);
void XRadar_cnn_stream_full_InterruptDisable(XRadar_cnn_stream_full *InstancePtr, u32 Mask);
void XRadar_cnn_stream_full_InterruptClear(XRadar_cnn_stream_full *InstancePtr, u32 Mask);
u32 XRadar_cnn_stream_full_InterruptGetEnabled(XRadar_cnn_stream_full *InstancePtr);
u32 XRadar_cnn_stream_full_InterruptGetStatus(XRadar_cnn_stream_full *InstancePtr);

#ifdef __cplusplus
}
#endif

#endif
