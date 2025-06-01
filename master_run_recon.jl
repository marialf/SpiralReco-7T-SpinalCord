#-----------------------------------------------------------------------------------
# [SpiralReco-7T-SpinalCord](@id master_run_recon)
#-----------------------------------------------------------------------------------
#=

#  Project: Master's Thesis – B0-corrected single-shot spiral MRI of the cervical spinal cord at 7 Tesla
#  Author: [Maria Leseth Føyen], NTNU, Department of Physics
#  Supervisor: [Johanna Vannesjö]
#  Date: [June 2025]


This code builds on the open-source GIRFReco.jl framework.
Script for running the full reconstruction pipeline for **single-slice** spinal cord data.
Performs both Cartesian reference reconstruction (including field map estimation 
and SENSE map calculation) and the subsequent spiral reconstruction.

This was the initial reconstruction script created for early-stage development and 
debugging. It remains separated from `multi_slice_run.jl` to reduce the risk of breaking 
the multi-slice pipeline. Ideally, the two should be merged in future work.
=#


#=
## 1. Setup

The necessary Julia packages needed for spiral reconstruction.
=#

#Our developed packages
using GIRFReco, MRIGradients

#MRIReco and its sub-packages
using MRIReco, FileIO, MRIFiles, MRIBase, MRICoilSensitivities

using RegularizedLeastSquares

using ImageTransformations

using Dierckx

using MosaicViews

using DelimitedFiles, FourierTools, ROMEO, Unitful, ImageView

import GIRFReco: load_map, save_map

using Plots, LaTeXStrings


data_root_project_path = "/Volumes/Master/ReconDirectory"

include("master_config_recon.jl")

# Todo: update to dynamic change of recon_size dependent on the reconstructed spiral
# Todo 2: 

# Check the data paths, for spiral file and save path for reconstruction
params_general[:scan_fullpath]
params_general[:recon_save_fullpath]
params_general[:mm_shift]
params_general[:recon_save_filename] 
params_general[:voxel_shift]


#Gradient reader file
include("/src/io/master_gradient_reader.jl")

# Utils files
include("/src/utils/master_utils_report.jl")
include("/src/utils/master_analyse_utils.jl")




### Configurations /necessary parameters to define 

num_total_diffusion_directions = params_general[:num_total_diffusion_directions]
## Determine to reconstruct single-interleave data, or one interleave out of multi-interleave data.
is_single_interleave = true #~(length(params_general[:scan_fullpath]) > 1)

start_idx_interleave = 1;
slice_selection = [1]

params_general[:b0_map_save_fullpath]
# Only run when coil/B0 maps have not been calculated
if ~(params_general[:do_load_maps] && isfile(params_general[:b0_map_save_fullpath]))
    @info "Running cartesian_recon to retrieve maps (cartesian_sensitivity and b0_maps)"
    run_cartesian_recon(params_general) 
end


# Load the SENSE maps from the previously calculated NIfTI files.
@info "Loading SENSE and B0 maps from $(params_general[:sensitivity_save_fullpath])"
cartesian_sensitivity = load_map((params_general[:sensitivity_save_fullpath]); do_split_phase = true)
b0_maps = load_map(params_general[:b0_map_save_fullpath])
num_slices = size(b0_maps, 3)
# Load the cartesian recon also, for plotting
cart_recon = load_map(params_general[:map_save_fullpath]; do_split_phase = true)

plot_reconstruction(cart_recon[:, :, 1, 1], 1:size(cart_recon, 3), b0_maps[:, :, 1], is_slice_interleaved = false, rotation = 0,savePlot = false, savePath = "")

### Parameters for the spiral data 

#############################################

reload_spiral_data = true; # Set true if we need to reload raw data compulsively.

# Check the frequency of the cartesian data
raw_cart = RawAcquisitionData(ISMRMRDFile(params_general[:map_scan_fullpath]))
frq_cart = raw_cart.params["H1resonanceFrequency_Hz"]


# Calculate slice index with spatially ascending order in the raw kspace data file.
raw_temp  = RawAcquisitionData(ISMRMRDFile(params_general[:scan_fullpath]))
slice_idx_array_spiral = get_slice_order(raw_temp, num_slices, (num_slices+1)*2, 2)

repetition_values = [profile.head.idx.repetition + 1 for profile in raw_temp.profiles]
println("Repetitions in raw data: ", unique(repetition_values))

frequency_spiral = raw_temp.params["H1resonanceFrequency_Hz"]
@info "The modulation frequency of the spiral scan is $frequency_spiral"


# If offset between modulation frequency of cartesian scan and spiral scan, add the offset to the B0 map globally
if params_general[:do_correct_with_b0_map] && !(frequency_spiral == frq_cart)
    rads_offset = (frequency_spiral - frq_cart)*2*pi
    b0_maps = b0_maps .+ rads_offset
    @info "Added offset of $rads_offset rad/s globally to the b0 map"
end

if isempty(slice_choice) || !(@isdefined slice_choice)
    slice_choice = collect(1:num_slices)
end

is_multislice = length(slice_choice) > 1

if !is_multislice
    selected_slice = slice_choice
else
    selected_slice = sort(vec(slice_choice))
end

#=
The first step is to select the part of spiral k-space data that we 
would like to reconstruct. This include selecting slices, diffusion directions, 
and averages that we want.
=#

params_spiral = Dict{Symbol,Any}() 
# Egne keys for trajektorie normalisering
params_spiral[:enc_fov] = (fov,fov,1) # Må spesifisere 1 på tredje aksen
params_spiral[:enc_size] = (eS,eS,1)
params_spiral[:resolution] = params_spiral[:enc_fov][1]/params_spiral[:enc_size][1]
#params_spiral[:crop_idx] = params_general[:crop_idx]
params_spiral[:recon_size] = Tuple(params_general[:recon_size]) #fov in read_nominal_gradient_file
params_spiral[:interleave] = start_idx_interleave
params_spiral[:num_samples] = size(raw_temp.profiles[slice_selection[1]].data, 1) #params_general[:num_adc_samples]
params_spiral[:delay] = 0.00000
# Full paths of raw k-space data files of spiral acquisition
params_spiral[:interleave_data_filenames] = params_general[:scan_fullpath] 
# Full paths of k-space trajectory txt file
params_spiral[:traj_filename] = params_general[:gradient_fullpath]
params_spiral[:do_multi_interleave] = !is_single_interleave
params_spiral[:do_odd_interleave] = false
params_spiral[:num_interleaves] = is_single_interleave ? 1 : length(params_spiral[:interleave_data_filenames]) # one interleaf per file, count files, if filenames are array of strings (not only one string)
# Set for single/ multiple slices
params_spiral[:single_slice] = true
params_spiral[:single_rep] = true # False when reconstructing all measurements
if params_spiral[:single_rep] == false
    params_spiral[:excitations] = collect(1:1:20) # = number of profiles (?)
else
    params_spiral[:excitations] = slice_selection
end



########
### Load the spiral data and adjust the sensititvy array (the adjustment is not really necessary considering we only are dealing with one slice here)

if reload_spiral_data || !(@isdefined imaging_acq_data)
    @info "Reading spiral data and merging interleaves"
    # Image acqusition data, trajectory calculation and merging with gradient file
    imaging_acq_data = my_merge_raw_interleaves(params_spiral,false) 
    b0_maps = b0_maps[:, :, invperm(slice_idx_array_spiral)]
    cartesian_sensitivity = cartesian_sensitivity[:, :, invperm(slice_idx_array_spiral), :]
    # Load the b0 maps only to be able to call the plotting function
end

# If we want to reconstruct all repetitions of the single-slice data, have to interpret slice dimension as repetitions. 
if params_spiral[:single_rep] == false
    # Permute dimensions to interpret the kdata as 20 slices( not 20 repetitions)
    imaging_acq_data.kdata = permutedims(imaging_acq_data.kdata, (1,3,2))
    num_repetitions, num_contrasts = numRepetitions(imaging_acq_data), numContrasts(imaging_acq_data)
    # Expand to 20 repetitions
    resized_b0_maps = repeat(resized_b0_maps, outer=(1, 1, 20))  # Replicate along the 3rd dimension
    # Expand to 20 repetitions
    sensitivity = repeat(sensitivity, outer=(1, 1, 20, 1))  # Replicate along the 3rd dimension
end



## Apply GIRFS, 1st order

girf1 = readGIRFFile(params_general[:girf_fullpath][1], params_general[:girf_fullpath][2], params_general[:girf_fullpath][3], "GIRF_FT",false)
girf_applier1 = GirfApplier(girf1, params_general[:gamma])

if params_general[:do_correct_with_girf_k1]
    @info "Correcting For GIRF"
    apply_girf!(imaging_acq_data, girf_applier1)
end

#Correct trajectory with the zeroth order GIRFs (K0)
girf_k0 = readGIRFFile(params_general[:girf_fullpath][1], params_general[:girf_fullpath][2], params_general[:girf_fullpath][3], "b0ec_FT", true)
girf_applier_k0 = GirfApplier(girf_k0, params_general[:gamma])

params_general[:do_correct_with_girf_k0]

if params_general[:do_correct_with_girf_k0]
    @info "Correcting For k₀"
    apply_k0!(imaging_acq_data, girf_applier_k0)
end

check_acquisition_nodes!(imaging_acq_data)
shift_kspace!(imaging_acq_data, params_general[:voxel_shift])

if params_general[:do_correct_with_girf_k0]
    non_corr_traj = non_corr_image_data.traj
    corr_traj = imaging_acq_data.traj
    params_spiral[:save_plot_path] = joinpath(params_general[:save_plot_path], "trajectory_diff_g1_and_k0.png")
    plot_kspace_trajectory_radm(non_corr_traj, corr_traj, params_spiral, 1, 1000)
end
#=
#### 3.2.5 Processing Coil Sensitivity Maps

We need to preprocess the coil sensitivity maps before reconstruction. 
This includes resizing the coil maps to the size of output encoding matrix size; 
compress the channels according to user's setting to achieve a faster reconstruction.
=#
sensitivity = mapslices(x -> imresize(x, params_spiral[:recon_size][1], params_spiral[:recon_size][2]), cartesian_sensitivity, dims = [1, 2])


if params_general[:do_correct_with_b0_map]
    resized_b0_maps = mapslices(x -> imresize(x, params_spiral[:recon_size][1], params_spiral[:recon_size][2]), b0_maps, dims = [1, 2])
end


#############################################

### Parameters for the spiral reconstruction 

#############################################
selected_slice = [1]

@info "Setting parameters for reconstruction"
params_recon = Dict{Symbol,Any}()
params_recon[:reco] = "multiCoil"
params_recon[:reconSize] = Tuple(params_spiral[:recon_size][1:2])# cannot avoid camel-case here since it is defined by MRIReco.jl and RegularizedLeastSquares.jl
params_recon[:regularization] = "L2"
params_recon[:λ] = 1e-3
params_recon[:iterations] = params_general[:num_recon_iterations]
params_recon[:solver] = "cgnr"
params_recon[:solverInfo] = SolverInfo(ComplexF32, store_solutions = false)
params_recon[:senseMaps] = ComplexF32.(sensitivity[:, :, selected_slice, :])
if params_general[:do_correct_with_b0_map]
    params_recon[:correctionMap] = ComplexF32.(-1im .* resized_b0_maps[:, :, selected_slice]) # cannot avoid camel-case here since it is defined by MRIReco.jl and RegularizedLeastSquares.jl
end

params_general[:do_correct_with_b0_map]
params_recon[:correctionMap]
#= 
Finally we can call reconstruction function of the package `MRIReco.jl` 
to perform final spiral image reconstruction.
=#
@info "Performing Spiral Reconstruction"
@time reco = reconstruction(imaging_acq_data, params_recon)
GC.gc() # Recommended to force triger garbage collection especially when encountering memory issues.


plot_reconstruction_simple(reco[:, :, 1], 3, 99.6; rotation=0)


if params_general[:do_save_recon]
    resolution_tmp = fieldOfView(imaging_acq_data)[1:2] ./ encodingSize(imaging_acq_data)
    resolution_mm = (resolution_tmp[1], resolution_tmp[2], fieldOfView(imaging_acq_data)[3] * (1 + params_general[:slice_distance_factor_percent] / 100.0)) #for 2D only, since FOV[3] is slice thickness then, but gap has to be observed
    save_map(
        "/Volumes/Master/ReconDirectory/Reconstructions/060525_scans/single_slice/192mm1p00mmR3_noSENSE_recon_2.nii",
        params_general[:saving_scalefactor] * reco.data[:, :, sortperm(invperm(slice_idx_array_spiral)[selected_slice])],
        resolution_mm;
        do_split_phase = true,
        do_normalize = params_general[:do_normalize_recon],
    )
    save_path = params_general[:recon_save_fullpath]
    @info "Spiral reconstruction saved to: $save_path"
end

