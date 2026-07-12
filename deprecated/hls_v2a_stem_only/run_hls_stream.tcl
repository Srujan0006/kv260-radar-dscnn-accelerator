open_project radar_cnn_stream_prj
source ../config/hardware_config.tcl
set_top $V2A_TOP_FUNCTION

add_files radar_cnn_stream.cpp
add_files radar_cnn_stream.h
add_files stream_types.h
add_files weights.h
add_files -tb tb_stream.cpp

open_solution "solution1" -flow_target vivado
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

exit
