# Vitis HLS script for radar 2D CNN accelerator.
# Usage from hls folder:
#   RADAR_CNN_HLS_TESTBENCH=tb_real_samples.cpp vitis-run --mode hls --tcl run_hls.tcl
#     -> real-sample C simulation only
#   vitis-run --mode hls --tcl run_hls.tcl
#     -> default testbench, synthesis, and IP export

open_project radar_cnn_hls_prj
source ../config/hardware_config.tcl
set_top $V1_TOP_FUNCTION

add_files radar_cnn_hls.cpp
add_files radar_cnn_hls.h
add_files weights.h
if {[info exists ::env(RADAR_CNN_HLS_TESTBENCH)]} {
    set tb_file $::env(RADAR_CNN_HLS_TESTBENCH)
} else {
    set tb_file "tb_radar_cnn_hls.cpp"
}
add_files -tb $tb_file

if {[info exists ::env(RADAR_CNN_HLS_MODE)]} {
    set run_mode $::env(RADAR_CNN_HLS_MODE)
} elseif {[info exists ::env(RADAR_CNN_HLS_TESTBENCH)]} {
    set run_mode "csim"
} else {
    set run_mode "synth"
}

open_solution "solution1" -flow_target vivado

# KV260/K26 SOM part. Adjust if your board flow uses a platform instead.
set_part $PROJECT_PART
create_clock -period $CLOCK_PERIOD_NS -name default

if {$run_mode eq "csim" || $run_mode eq "all"} {
    if {$tb_file eq "tb_real_samples.cpp"} {
        set sample_file [file normalize "../reports/fixed_point_validation/input_samples.txt"]
        csim_design -argv $sample_file
    } else {
        csim_design
    }
}

if {$run_mode eq "synth" || $run_mode eq "all"} {
    csynth_design
    if {$run_mode eq "all"} {
        cosim_design
    }
    export_design -format ip_catalog
}

exit
