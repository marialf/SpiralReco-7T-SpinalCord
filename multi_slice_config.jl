## New file for multiple slices reconstruction configuration - to avoid ruining something in the single-slice config file. 
##Should ideally be unified with the single-slice configuration 

## This recon_config.jl file describes all reconstruction parameters, as well as data locations and selections for an iterative non-Cartesian reconstruction that relies 
#  on an external reference scan (Cartesian) to estimate calibration maps (coil sensitivities, B0 maps)

using Dates

### SET SPIRAL PARAMS SPECIFIC FOR SPIRAL SEQENCE AND RECONSTRUCTION

fov = 192  # Field of view [mm]
spiral_res = 1.00  # Resolution [mm]
R = 3 # Acceleration factor
mm_shift = 60
idx_av = 1

voxel_shift = mm_shift/spiral_res # Calculate the shift in voxels for the spiral sequence
# Reconoption
recon_res = spiral_res # Default; spiral resolution
recon_fov = 192

rel_res = recon_res/spiral_res

# Calculate encoding size from resolution and fov
# NB: must be an even number
if iseven(round(fov/recon_res))
    eS = Int64(round(fov/recon_res))
else
    eS = Int64(round(fov/recon_res)) - 1
end
    # Calculate recon size from resolution and fov
    # NB: must be an even number
if iseven(round(recon_fov/recon_res))
    rS = Int64(round(recon_fov/recon_res))
else
    rS = Int64(round(recon_fov/recon_res)) - 1
end

###
# Find names of files
(fres, ires) = modf(spiral_res)
fres = Int16(round(fres, digits=2)*100)
ires = Int16(ires)

current_spiral = "$(fov)mm$(ires)p$(lpad(fres, 2, '0'))mmR$(R)"

recon_id =  "may_mult_slice" #"april_250320_mult_slice"# "60325_scans"# For multi-slice runs:  # For offset runs: "DIFF_OFFSETS_IV_SC"*spiral "IV_SC_140mm_1p71_R1" #

params_general = Dict{Symbol,Any}()
# Gyromagnetic ratio, in unit of Hz
params_general[:gamma] = 42577478;

## General options for recon script
params_general[:do_load_maps] =  true # if true, reloads B0/SENSE maps instead of recalculating
params_general[:do_save_recon] = true          # if true, saves reconstruction and all auxiliary image data (maps) as NIfTI files
params_general[:do_plot_recon] = false         # if true, plots intermediate debugging and output recon figures (needs graphics, not recommended in multi-thread mode due to PyPlot)
params_general[:do_process_map_scan] = true         # if true, compute sensitivity and B0 maps from reconstructed Cartesian scan   
params_general[:do_save_processed_map_scan] = false; # save ISMRMD file of preprocessed Cartesian data (before recon)

## Reconstruction Parameters
# update time stamp for new recon, otherwise keep fixed, will create a new recon/<recon_id> directory
#params_general[:recon_id] = Dates.format(Dates.now(), "yyyy-mm-dd_HH_MM_SS") # recon ID is recon_id
# params_general[:recon_id] = "2022-10-20_09_07_07"
params_general[:recon_id] = recon_id; # Namme of folder with results
params_general[:do_correct_with_b0_map] = true
params_general[:do_correct_with_girf_k1] = true
params_general[:do_correct_with_girf_k0] = false
params_general[:num_virtual_coils] = 0;
params_general[:do_coil_compression] = false;
params_general[:mm_shift] = mm_shift # [mm] 10 for phantom head, 60 for sc_140mm_1p71
params_general[:voxel_shift] = [0, -voxel_shift]
params_general[:res_cartesian] = 2 # [mm]
#params_general[:fov_shift] = [0, -5]; # Unit: number of voxels (mmshift /resolution = 10 mm/2 mm = 5) [0,-5] for phantom data hc, [0,-30] for IV data sc
#params_general[:crop_idx] = crop_idx # Fra Marens Matlab kode, unik for hver spiral

## Scan parameters, Additional acquisition information, e.g., slice distance etc.

#Total number of ADC points BEFORE the rewinder at the end of the spiral readout.
# = params_general[:crop_idx]*4 # Crop_idc times ratio time step gradient/time step sampling
params_general[:recon_size] = [rS, rS, 1] 
params_general[:num_recon_iterations] = 10; # number of recon iterations (for both Cartesian and Spiral recon)
params_general[:b0_map_beta] = 0.01 # for estimate_b0_maps, * `β` - Regularization parameter controlling roughness penalty (larger = smoother, default 5e-4)
params_general[:do_normalize_recon] = false # set max abs to 1
params_general[:saving_scalefactor] = 1.0e9 # 1 # typical range of recon intensities is 1e-7, rescale when saving, e.g., to 0...1000 roughly for fMRI analysis
params_general[:num_total_diffusion_directions] = 0;          # Need to specify total diffusion directions included in the raw data
params_general[:slice_distance_factor_percent] = 50 # Scan parameters, Additional acquisition information, e.g., slice distance etc.

# Data selector
#  Choose diffusion direction; starting from 0 (b=0) to the total number in MDDW protocol, e.g. for 6 diffusion directions, 1-6 stands for 6 DWIs)
# boolean is_called_from_global_recon is true, if this RunReconLoop is active
# If is_called_from_global_recon is false or not defined, the data selector needs to be defined here.
if !(@isdefined is_called_from_global_recon) || !is_called_from_global_recon
    global selector = Dict{Symbol,Any}()
    selector[:avg] = 1
    selector[:seg] = 1
    selector[:dif] = 0
end

#=
### Specifying Directories
=#
params_general[:spiral] = current_spiral
params_general[:project_path] = data_root_project_path # Root path for the project


params_general[:save_plot_path] = joinpath(params_general[:project_path], "Reconstructions", "Plots")
#Path to ISMRMRD files (raw k-space data) [Input]
params_general[:data_path] = joinpath(params_general[:project_path], "Data")


params_general[:spiral_path] = joinpath(params_general[:data_path], "Spirals")

#Path to spiral readout gradient files [Input]
params_general[:gradients_path] = joinpath(params_general[:data_path], "Gradients")
#Path to GIRF files [Input]
params_general[:girf_path] = joinpath(params_general[:data_path], "GIRF")

#Path to middle results (coil and B₀ maps) files [Output], Common for all with the current recon ID (same cartesian field map scan)
params_general[:results_path] = joinpath(params_general[:project_path], "Reconstructions", params_general[:recon_id])
#Path to final reconstructed spiral images [Output], specific for certain spiral 
params_general[:recon_save_path] = joinpath(params_general[:project_path],"Reconstructions", params_general[:recon_id]) # "DiffOffsets_$(fov)_R$R" #all_measurements_$(fov)R$R") #"Offset_report_first_slice"

#= Change save path according to applied corrections



if (params_general[:do_correct_with_girf_k1])
    params_general[:recon_save_path] = params_general[:recon_save_path]*"_"*string(params_general[:b0_map_beta])*"_girf1"
end

if (params_general[:do_correct_with_girf_k0])
    params_general[:recon_save_path] = params_general[:recon_save_path]*"_k0"
end

if !(recon_res == spiral_res)
    params_general[:recon_save_path] = joinpath(params_general[:results_path], "recon_res_$rel_res")
end
=#

#=
### Specifying File Names
=#

## Input files: 

# Map scan file (Cartesian multi-echo file)
params_general[:map_scan_filename] = "Fieldmaps/ms_MID00059_fm_2mm_magph_angulated.h5"#vol2_ms_MID00036_fm_2mm_magph.h5" #ms_MID00059_fm_2mm_magph_angulated.h5" #  vol2_ms_MID00036_fm_2mm_magph.h5" , ms_MID00036_fm_2mm_magph.h5 #"Fieldmaps/lTE_ms_MID00053_fm_2mm_magph.h5"# 250320 first spirals "Fieldmaps/ms_MID00036_fm_2mm_magph.h5" # 60325 scans :inVivo_fieldmap_prescan.h5 #"Fieldmaps/32ch_phantom_fm_converted.h5"# # Cartesian dual-echo file, for coil and B₀ maps calculation [Input]
params_general[:map_scan_filename_stem] = "ms_MID00059_fm_2mm_magph_angulated.h5" #"vol2_ms_MID00036_fm_2mm_magph.h5" #"32ch_phantom_fm_converted.h5" #
params_general[:mapTEs_ms] = [4.08, 5.1] 
 # File name for the spiral gradient [Input] 
params_general[:gradient_filename] = joinpath("ArbGradientRO_$current_spiral.txt")
#  non-Cartesian (Spiral) scan file: 
#params_general[:scan_filename] = ["Spirals/sp_140_inVivo_sc.h5"] # ISMRMRD Raw k-space data for spiral acquisition [Input]

#GIRFS:
params_general[:girf_filename] = ["GIRF_cross_x.mat", "GIRF_cross_y.mat", "GIRF_cross_z.mat"] 

#Output files
params_general[:scan_filename_stem] = "$(current_spiral)" #"32ch_140mm1p71mmR1.h5"# # Main file name when saving the result #angulated_ foran, hvis vinkel
params_general[:processed_map_scan_filename] = "preprocessed_$current_spiral" #"preprocessed_sp_140_inVivo_sc.h5" # file name for preprocessed data (remove oversampling, permute dimensions wrt MRIReco) [Output]
params_general[:map_save_filename] = splitext(params_general[:map_scan_filename_stem])[1] * "_angulated_reconmap.nii" # File name for reconstructed dual-echo Cartesian images [Output]
params_general[:sensitivity_save_filename] = splitext(params_general[:map_scan_filename_stem])[1] * "_angulated_sensemap.nii" # File name for calculated coil sensitivity maps [Output]
params_general[:b0_map_save_filename] = string(params_general[:b0_map_beta])*"_angulated_b0map.nii"; splitext(params_general[:map_scan_filename_stem])[1] * "_"* string(params_general[:b0_map_beta])*"_b0map.nii"; # File name for calculated off-resonance (B₀) maps [Output] # _shift_1.6.

#=
File name for the final reconstructed spiral image.
If we reconstructing multiple spiral data files (e.g. multiple interleaves) through `RunReconLoop.jl`, 
the file name for the final reconstructed image is concatenated from multiple scan file names. 
Otherwise, just append `_recon.nii` as suffix to file name.
=#
if (params_general[:do_correct_with_b0_map])
    params_general[:recon_save_filename] = params_general[:scan_filename_stem]*"_"*string(params_general[:b0_map_beta])*"wb0"
else
    params_general[:recon_save_filename] = params_general[:scan_filename_stem]*"_nob0_noSENSE"
end




# otherwise, just concat _recon.nii to file name
params_general[:recon_save_filename] = params_general[:recon_save_filename] * "angulated_recon.nii"


#=
### Assembling Full Paths

Assembling directories and file names for final full pathes. 
These are automated operations.
=#
params_general[:gradient_fullpath] = joinpath(params_general[:gradients_path], params_general[:gradient_filename]) # Full paths of spiral readout gradients
params_general[:girf_fullpath] = joinpath.(params_general[:girf_path], params_general[:girf_filename]) # Full paths of GIRF files
params_general[:map_scan_fullpath] = joinpath(params_general[:data_path], params_general[:map_scan_filename]) # Full path of dual-echo Cartesian data
params_general[:scan_fullpath] = joinpath.(params_general[:spiral_path], "ms_angulated_"*params_general[:scan_filename_stem]*".h5") # Full paths of raw k-space data files of spiral acquisition. "ms_"* for multi-slice


params_general[:processed_map_scan_fullpath] = joinpath(params_general[:recon_save_path], params_general[:processed_map_scan_filename]) # Full paths of pre-processed Cartesian dual-echo data [Output]
params_general[:recon_save_fullpath] = joinpath(params_general[:recon_save_path], params_general[:recon_save_filename]) # Full paths of the reconstructed spiral image [Output]
params_general[:map_save_fullpath] = joinpath(params_general[:recon_save_path], params_general[:map_save_filename]) # Full paths of reconstructed dual-echo Cartesian images [Output]
params_general[:sensitivity_save_fullpath] = joinpath(params_general[:recon_save_path], params_general[:sensitivity_save_filename]) # Full paths of calculated coil sensitivity maps [Output]
params_general[:b0_map_save_fullpath] = joinpath(params_general[:recon_save_path], params_general[:b0_map_save_filename]); # Full paths of calculated off-resonance (B₀) maps [Output]




#=
## Final Steps

Optional: If the path for results writing is not existing, create it.

As the last step of configuration, copy this config file 
to the recon path for further checking and debugging purposes.
=#


if ~ispath(params_general[:recon_save_path])
    mkpath(params_general[:recon_save_path])
end

# copies this config file to the recon path for later checks of parameter functions
cp(@__FILE__, joinpath(params_general[:recon_save_path], "recon_config.jl"); force = true)

