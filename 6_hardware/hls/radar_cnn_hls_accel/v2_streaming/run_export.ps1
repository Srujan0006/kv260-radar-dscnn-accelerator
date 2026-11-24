# Exports the IP and runs standalone place-and-route (Vitis HLS 2022.2).
Set-Location $PSScriptRoot
& 'C:\Xilinx\Vitis_HLS\2022.2\bin\vitis_hls.bat' -f run_export.tcl *> export_console.log
