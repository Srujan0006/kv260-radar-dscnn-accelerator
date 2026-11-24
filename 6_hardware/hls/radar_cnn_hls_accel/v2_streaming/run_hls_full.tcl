open_project radar_cnn_stream_prj
# The shared config lives at radar_cnn_hls_accel/config/ (one level up).
# The fallback path supports running from a nested copy.
if {[file exists ../config/hardware_config.tcl]} {
    source ../config/hardware_config.tcl
} else {
    source ../../config/hardware_config.tcl
}

set_top $V2C_TOP_FUNCTION

add_files radar_cnn_stream.cpp
add_files radar_cnn_stream.h
add_files stream_types.h
add_files stream_layers.h
add_files hls_stream_compat.h
add_files weights.h
add_files -tb tb_stream.cpp

# Separate solution so the V2A (solution1) and V2B (solution2_dw1)
# reports stay untouched.
open_solution "solution3_full" -flow_target vivado
set_part $PROJECT_PART
create_clock -period $CLOCK_PERIOD_NS -name default

if {[info exists ::env(RADAR_CNN_HLS_MODE)]} {
    set run_mode $::env(RADAR_CNN_HLS_MODE)
} else {
    set run_mode "all"
}

if {$run_mode eq "csim" || $run_mode eq "all"} {
    csim_design
}
if {$run_mode eq "synth" || $run_mode eq "cosim" || $run_mode eq "all"} {
    csynth_design
}
if {$run_mode eq "cosim" || $run_mode eq "all"} {
    cosim_design
}

if {$run_mode eq "all"} {
    export_design -format ip_catalog
}

exit
