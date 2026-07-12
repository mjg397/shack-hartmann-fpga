set script_dir [file dirname [file normalize [info script]]]
cd $script_dir

if {![info exists part_name]} {
    set part_name "xc7a100tcsg324-1"
}

create_project shwfs_vivado $script_dir/project -part $part_name -force

add_files -fileset sources_1 [glob $script_dir/*.v]
add_files -fileset sources_1 [glob $script_dir/data/*.hex]
set_property top shwfs_vivado_ila_top [current_fileset]
set_property verilog_define {USE_ILA} [current_fileset]

create_ip -name ila -vendor xilinx.com -library ip -module_name ila_0
set_property -dict {
    CONFIG.C_NUM_OF_PROBES {1}
    CONFIG.C_PROBE0_WIDTH {640}
    CONFIG.C_DATA_DEPTH {1024}
} [get_ips ila_0]
generate_target all [get_ips ila_0]

update_compile_order -fileset sources_1
