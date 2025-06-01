#-----------------------------------------------------------------------------------
# [SpiralReco-7T-SpinalCord](@id multi_slice_run)
#-----------------------------------------------------------------------------------
#=
This code builds on the open-source GIRFReco.jl framework.
Script for running the full reconstruction pipeline for **multi-slice** spinal cord data.
Includes Cartesian reference reconstruction (field map estimation and SENSE maps) and
spiral image reconstruction for multiple slices.

Developed alongside `master_run_recon.jl` (single-slice) during this thesis. 
A unified pipeline could be considered in the future to streamline usage.
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


Plots.default(
     guidefontsize = 16,  # For axis labels (guides)
     tickfontsize = 14,   # For axis tick labels
     legendfontsize = 14
)


# Data root path 
data_root_project_path = "/Volumes/Master/ReconDirectory"

#Gradient reader file
include("/src/io/master_gradient_reader.jl")

# Utils files
include("/src/utils/master_utils_report.jl")
include("/src/utils/master_analyse_utils.jl")



#=
Choose which diffusion directions and averages to be processed. 
Diffusion direction index starts from 0 (b=0) to the total number in MDDW protocol (e.g. for 6 diffusion directions, 1-6 stands for 6 DWIs). 
Index for average starts from 1.
=#
diffusion_direction = 0
idx_average = 1

num_total_diffusion_directions = params_general[:num_total_diffusion_directions]
## Determine to reconstruct single-interleave data, or one interleave out of multi-interleave data.
is_single_interleave = true #~(length(params_general[:scan_fullpath]) > 1)

start_idx_interleave = 1;
# Leavve empty to select all slices
slice_choice = [1,7,10,15]
params_general[:b0_map_save_fullpath]

# Only run when coil/B0 maps have not been calculated
if ~(params_general[:do_load_maps] && isfile(params_general[:b0_map_save_fullpath]))
    @info "Running cartesian_recon to retrieve maps (cartesian_sensitivity and b0_maps)"
    run_cartesian_recon(params_general, doMultPi = false) 
end


# Check the slice position of the cartesian data
raw_cart = RawAcquisitionData(ISMRMRDFile(params_general[:map_scan_fullpath]))
frq_cart = raw_cart.params["H1resonanceFrequency_Hz"]


params_general[:b0_map_save_fullpath]

# Load the SENSE maps from the previously calculated NIfTI files.
@info "Loading SENSE and B0 maps from $(params_general[:sensitivity_save_fullpath])"
cartesian_sensitivity = load_map((params_general[:sensitivity_save_fullpath]); do_split_phase = true)
b0_maps = load_map(params_general[:b0_map_save_fullpath])
num_slices = size(b0_maps, 3)
# Load the cartesian recon also, for plotting
cart_recon = load_map(params_general[:map_save_fullpath]; do_split_phase = true)

plot_slices(cart_recon[:,:,:,1], "First echo images, angulated")
slice_indices_plot = [1, 7, 10, 15]
plot_multiple_b0_maps(b0_maps[:,:,slice_indices_plot], doSet_scale=true)

#plot_reconstruction(cart_recon[:, :, :, 1], 1:size(cart_recon, 3), b0_maps[:, :, :], is_slice_interleaved = false, rotation = 0,savePlot = false, savePath = "")

for i in(1:15)
    plot_single_b0_map(b0_maps[:,:,i], "0.05, slice: $i", doSet_scale = false, doOffset = false, rad_s_offset = 0)
end

# Get the order of slices from the RawAcqData header
slice_idx_array_cartesian = get_slice_order(raw_cart, num_slices, 1, 1)
# Check the z-position of my slices:


println(slice_idx_array_cartesian)
############################################# SPIRAL RECONSTRUCTION ####################


reload_spiral_data = true; # Set true if we need to reload raw data compulsively.
#=
### 3.2 Preparation of Spiral Reconstruction

With off-resonance (B₀) maps and coil sensitivity maps calculated, 
before the reconstruction of spiral images, there are necessary steps to prepare for 
the related data. 

#### 3.2.1 Data Selection

The first step is to select the part of spiral k-space data that we 
would like to reconstruct. This include selecting slices, diffusion directions, 
and averages that we want.

First we sort the slice index that we selected to reconstruct.
=#

if isempty(slice_choice) || !(@isdefined slice_choice)
    slice_choice = collect(1:num_slices)
end

is_multislice = length(slice_choice) > 1

if !is_multislice
    selected_slice = slice_choice
else
    selected_slice = sort(vec(slice_choice))
end


# Calculate slice index with spatially ascending order in the raw kspace data file.
raw_temp  = RawAcquisitionData(ISMRMRDFile(params_general[:scan_fullpath]))


# Get the correct order of the slices (from negative to positive max position)
slice_idx_array_spiral = get_slice_order(raw_temp, num_slices, num_slices+2, 2)
println(slice_idx_array_spiral)

repetition_values = [profile.head.idx.repetition + 1 for profile in raw_temp.profiles]
println("Repetitions in raw data: ", unique(repetition_values))



#=
Next we select the data we would like to reconstruct from the ISMRMRD file. 

The ISMRMRD data are stored in the following loops:

Slice 1, Slice 2 ... Slice N   Slice 1, Slice 2 ... Slice N     Slice 1, Slice 2 ... Slice N ... 

|______ Diff Dir 1 ______|   |______ Diff Dir 2 ______| ... |______ Diff Dir N ______| ... 

|_________________________________ Average 1 ___________________________________| ... |___ Average N___| 

Here we chose the set corresponding to the b-value = 0 images under the first average as the example.

Note that (1) The raw data file begins with a series of pre-scan profiles with a length of `num_slices*2` 
and we want to skip them; (2) There is a B1 measurement data profile between each k-space readout profile 
which also need to be skipped. Thus the reading of data profiles starts from `num_slices*2 + 2` with 
an increment of `2`.

MARIA: or in our case, profiles from numSlices*1 + 2?
=#

excitation_list = collect(num_slices*1+2:2:num_slices*4) .+ diffusion_direction * num_slices * 2 .+ (idx_average - 1) * num_slices * (num_total_diffusion_directions + 1) * 1
slice_selection = excitation_list[selected_slice]

print(slice_selection[slice_idx_array_spiral[2]])
### Checking z-position with cartesian scan
z_pos_cartesian = zeros(num_slices)
for i in 1:num_slices
    # Assuming Cartesian raw was loaded as raw_cart
    z_pos_cartesian[i] = raw_cart.profiles[i].head.position[3]
end

z_pos_cartesian
# The slices sorted in ascending z position ( how the B0 maps and sensitivity maps are returned)
sorted_z_cartesian = z_pos_cartesian[slice_idx_array_cartesian]

# Also, get the z-positions after the maps are reordered to spiral order
z_pos_from_maps = sorted_z_cartesian[invperm(slice_idx_array_spiral)]
if sortperm(z_pos_from_maps) == slice_idx_array_spiral
    @info " Ordering of slices in the maps and spiral data align. "
else
    @info " Ordering of slices in the maps and spiral data does not align."
end

for i in 1:num_slices
    current_slice = slice_selection[slice_idx_array_spiral[i]]
    println("Z position of slice $(slice_idx_array_spiral[i]): ", raw_temp.profiles[current_slice].head.position[3])
end






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
params_spiral[:single_slice] = false
params_spiral[:single_rep] = true # When reconstructing all measurements
params_spiral[:excitations] = slice_selection


frequency_spiral = raw_temp.params["H1resonanceFrequency_Hz"]
@info "The modulation frequency of the spiral scan is $frequency_spiral"

# If offset between modulation frequency of cartesian scan and spiral scan, add the offset to the B0 map globally
if params_general[:do_correct_with_b0_map] && !(frequency_spiral == frq_cart)
    rads_offset = (frequency_spiral - params_general[:freq_cartesian])*2*pi
    b0_maps = b0_maps .+ rads_offset
    @info "Added offset of $rads_offset rad/s globally to the b0 map"
end



########
### Load the spiral data and adjust the sensititvy array


if reload_spiral_data || !(@isdefined imaging_acq_data)
    @info "Reading spiral data and merging interleaves"
    # Image acqusition data, trajectory calculation and merging with gradient file
    imaging_acq_data = my_merge_raw_interleaves(params_spiral,false) 
    cartesian_sensitivity = cartesian_sensitivity[:, :, invperm(slice_idx_array_spiral), :]
    # Load the b0 maps only to be able to call the plotting function
end


# Store a deep copy of the non-corrected image data, for later plotting against GIRF corrected
#non_corr_image_data = deepcopy(imaging_acq_data)
#check_acquisition_nodes!(non_corr_image_data)
#shift_kspace!(non_corr_image_data, params_general[:fov_shift])

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


@info "Setting parameters for reconstruction"
params_recon = Dict{Symbol,Any}()
params_recon[:reco] = "multiCoil" #"multiCoil"
params_recon[:reconSize] = Tuple(params_spiral[:recon_size][1:2])# cannot avoid camel-case here since it is defined by MRIReco.jl and RegularizedLeastSquares.jl
params_recon[:regularization] = "L2"
params_recon[:λ] = 1e-3
params_recon[:iterations] = params_general[:num_recon_iterations]
params_recon[:solver] = "cgnr"
params_recon[:solverInfo] = SolverInfo(ComplexF32, store_solutions = false)
params_recon[:senseMaps] = ComplexF32.(sensitivity[:, :, selected_slice, :])
params_recon[:correctionMap] = ComplexF32.(-1im .* resized_b0_maps[:, :, selected_slice])

@info "Running reconstruction"
@time reco = reconstruction(imaging_acq_data, params_recon)
GC.gc() 

params_general[:recon_save_fullpath]
reco.data[:, :, sortperm(invperm(slice_idx_array_spiral)[selected_slice]),1,:,1]
if params_general[:do_save_recon]
    #resolution_tmp = fieldOfView(imaging_acq_data)[1:2] ./ encodingSize(imaging_acq_data)
    resolution_mm = (1.0, 1.0, 3.0) # fieldOfView(imaging_acq_data)[3] * (1 + params_general[:slice_distance_factor_percent] / 100.0)) #for 2D only, since FOV[3] is slice thickness then, but gap has to be observed
    save_map(
        params_general[:recon_save_fullpath],
        params_general[:saving_scalefactor] * params_general[:saving_scalefactor] * reco.data[:, :, sortperm(invperm(slice_idx_array_spiral)[selected_slice]),1,:,1], # Endret fra reco.data[:,:,slice_choice] pga feilmelding
        resolution_mm;
        do_split_phase = true,
        do_normalize = params_general[:do_normalize_recon],
    )
    save_path = params_general[:recon_save_fullpath]
    @info "Spiral reconstruction saved to: $save_path"
end

#=
# Offsets:
# Define offsets in Hz, then convert to rad/s
fm_offsets_Hz = 20 #collect(5:5:45) # Offset range in Hz
fm_offsets = fm_offsets_Hz .* (2 * π)  # Convert to rad/s

# Generate offset titles
offset_title = ["$(fm_offsets_Hz[i]) Hz ($(round(fm_offsets[i], digits=2)) rad/s)" for i in eachindex(fm_offsets)]
joinpath(params_general[:recon_save_path], "$(current_spiral)_slice_$(slice_choice[1])_$(fm_offsets_Hz[1])Hz.nii")

doPlot = false
doSave = true


for i in (1:length(fm_offsets))
    current_offset = fm_offsets[i]
    title = offset_title[i]
    @info "Applying offset of $current_offset rad/s to the fieldmap."
    b0_map_offset = b0_maps .+ current_offset #apply_offset_and_plot(b0_map_original, current_offset)
    b0_maps_temp = b0_map_offset[:, :, invperm(slice_idx_array_spiral)]
    resized_b0_map_temp = mapslices(x -> imresize(x, params_spiral[:recon_size][1], params_spiral[:recon_size][2]), b0_maps_temp, dims = [1, 2])

    if !(params_spiral[:single_rep])
        resized_b0_map = repeat(resized_b0_map, outer=(1, 1, 20))  # Replicate along the 3rd dimension
    end

    
    params_recon[:correctionMap] = ComplexF32.(-1im .* resized_b0_map_temp[:, :, selected_slice])

    @info "Running reconstruction with field-map offset $title"
    @time reco = reconstruction(imaging_acq_data, params_recon)
    GC.gc() # Recommended to force triger garbage collection especially when encountering memory issues.
    # Extract the vertical center line profile
    #img = abs.(reco[:, :, 1, 1, 1, 1])  # Get magnitude image
    #img = img ./ maximum(img)  # Normalize

    #img_height, img_width = size(img)
    #center_col = img_width ÷ 2 - 2 # Middle column for extraction

    #line_profile = img[:, center_col]  # Extract vertical intensity profile
    #line_profiles[Symbol(current_offset)] = line_profile  # Store profile

    #recon_dict[Symbol(current_offset)] = reco
    #fm_dict[Symbol(current_offset)] = b0_map_offset

    if doPlot
        @info "Plotting reconstruction"
        plotlyjs(size=(1000, 1000))
        plot_reconstruction(
            recon_dict[Symbol(current_offset)][:, :, 1, 1, 1, 1],
            1:length(selected_slice),
            resized_b0_map,
            is_slice_interleaved = false,
            rotation = 0,
            savePlot = false,
            savePath = "",
            title_mag = "B0 recon, β = 0.01, offset = $off_title")
    end

    save_title = joinpath(params_general[:recon_save_path], "current_spiral_$(slice_choice[1])_$(fm_offsets_Hz[i])Hz.nii")
    println("Savetitle is ", save_title)

    if doSave#params_general[:do_save_recon]
        resolution_tmp = fieldOfView(imaging_acq_data)[1:2] ./ encodingSize(imaging_acq_data)
        resolution_mm = (1.0, 1.0, fieldOfView(imaging_acq_data)[3] * (1 + params_general[:slice_distance_factor_percent] / 100.0)) #for 2D only, since FOV[3] is slice thickness then, but gap has to be observed
        save_map(
            save_title,
            params_general[:saving_scalefactor] * reco.data[:, :, selected_slice],
            resolution_mm;
            do_split_phase = true,
            do_normalize = params_general[:do_normalize_recon],
        )
    end


end
=#


#=
if params_general[:do_correct_with_b0_map]
    params_recon[:correctionMap] = ComplexF32.(-1im .* resized_b0_maps[:, :, selected_slice]) # cannot avoid camel-case here since it is defined by MRIReco.jl and RegularizedLeastSquares.jl
end


#= 
Finally we can call reconstruction function of the package `MRIReco.jl` 
to perform final spiral image reconstruction.
=#
@info "Performing Spiral Reconstruction"
@time reco = reconstruction(imaging_acq_data, params_recon)
GC.gc() # Recommended to force triger garbage collection especially when encountering memory issues.

plot_single_b0_map(resized_b0_maps[:,:,1], 0.05)

plot_reconstruction_simple(reco[:,:,8,1,:,1], 3, 99.6; rotation=0)




#=

