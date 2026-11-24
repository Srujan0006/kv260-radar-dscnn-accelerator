# Runs RTL co-simulation from this folder (Vitis HLS 2022.2).
Set-Location $PSScriptRoot
$env:RADAR_CNN_HLS_MODE = 'cosim'
& 'C:\Xilinx\Vitis_HLS\2022.2\bin\vitis_hls.bat' -f run_hls_full.tcl *> cosim_console.log
