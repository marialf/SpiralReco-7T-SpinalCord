#------------------------------------------------------------------------------------
#  [SpiralReco-7T-SpinalCord](@id spiral_utils)
#------------------------------------------------------------------------------------
#
#  File: master_analyse_utils.jl
#  Project: Master's Thesis – B0-corrected single-shot spiral MRI of the cervical spinal cord at 7 Tesla
#  Author: [Maria Leseth Føyen], NTNU, Department of Physics
#  Supervisor: [Johanna Vannesjö]
#  Date: [June 2025]
#
#  Description:
#  Utility functions for data analysis, plotting, and evaluation related to
#  the master's thesis on high-resolution single-shot spiral MRI reconstruction
#  of the cervical spinal cord at 7 Tesla.
#
#  This code builds on and extends the open-source GIRFReco.jl framework.
#  Repository: https://github.com/YourUsername/SpiralReco-7T-SpinalCord
#
#------------------------------------------------------------------------------------


using Printf
using Plots, PlotlyJS
using Images, ImageUtils
using StatsBase
using ColorSchemes
using Statistics
using MIToS.Information
using ImageTransformations

"""
    crop_image_by_mm(img::Array{<:Complex}, crop_mm::Float64; fov_mm::Float64=192.0)

Crop a complex-valued image by a fixed number of mm on all sides. For plotting in the report, to reduce the amount of unneccesary black space around the object. 
Assumes img is of size [N_x, N_y, N_slices, 1].

Returns the cropped complex image.
"""
function crop_image_by_mm(img, crop_mm_x::Float64, crop_mm_y ::Float64 ; fov_mm::Float64=192.0)
    size_x, size_y = size(img)  #n_slices, _

    # Calculate pixel size in mm
    pixel_size_x = fov_mm / size_x
    pixel_size_y = fov_mm / size_y

    # Convert desired crop from mm to number of pixels
    crop_px_x = round(Int, crop_mm_x / pixel_size_x)
    crop_px_y = round(Int, crop_mm_y / pixel_size_y)

    # Define cropping range
    x_range = (crop_px_x + 1):(size_x - round(Int,crop_px_x/2))
    y_range = (crop_px_y + 1):(size_y - crop_px_y)

    # Apply cropping
    cropped_img = img[x_range, y_range, :, :]
    @info "x_range is : $x_range and y_range: $y_range"
    return cropped_img
end

"""
Extract a fixed region defined in physical mm from reconstructions of different resolutions.

# Arguments
- `resolution`: Pixel size in mm of the target image (e.g. 0.74 mm).
- `ref_resolution`: Resolution in mm for which `x_range` and `y_range` are originally defined (default 1.0).
- `x_range`, `y_range`: Index ranges in the reference image (assumed to be at `ref_resolution`).

# Returns
- Scaled `x_range`, `y_range` for the target resolution
"""
function extract_scaled_roi(resolution::Float64, x_range::UnitRange{Int}, y_range::UnitRange{Int}; ref_resolution::Float64 = 1.00)
    # Convert reference indices to physical mm
    x_mm = (x_range .- 1) .* ref_resolution
    y_mm = (y_range .- 1) .* ref_resolution

    # Convert mm back to pixel indices for new resolution
    x_idx = round.(Int, x_mm ./ resolution .+ 1)
    y_idx = round.(Int, y_mm ./ resolution .+ 1)

    return minimum(x_idx):maximum(x_idx), minimum(y_idx):maximum(y_idx)
end

"""
    apply_windowing(img::AbstractArray, lower_perc::Real, upper_perc::Real) -> Array

Window‐level an image to the intensity range between the given lower and upper
percentiles, then scale to [0, 1].

# Arguments
- `img`: Numeric array (magnitude image).
- `lower_perc`: Lower percentile (e.g. `3` ⇒ 3 %).
- `upper_perc`: Upper percentile (e.g. `99.6` ⇒ 99.6 %).

# Returns
- Array of the same size as `img`, values clipped to [0, 1].
"""
function apply_windowing(image, lower_perc, upper_perc)
    lo = percentile(vec(image), lower_perc)
    hi = percentile(vec(image), upper_perc)
    return clamp.((image .- lo) ./ (hi - lo), 0, 1)
end


"""
    plot_phase_difference(reconstruction::Array{<:Complex, 4}, title::String)

Plot phase difference between two complex image volumes (e.g., reconstructions).
Displays x = angle(conj(y) * z) for each slice in a subplot grid.

# Arguments
- `reconstruction`: Array of size `[Nx, Ny, num_slices, 2]`. Last dim holds two images.
- `title`: Plot title.

# Output
- Displays phase difference heatmaps for each slice.
"""


function plot_phase_difference(reconstruction, title)
    num_slices = size(reconstruction, 3)  # Number of slices
    if num_slices == 15
        rows, cols = 3,5 # Define subplot grid
        p = Plots.plot(layout=(rows, cols), size=(2500, 1600), plot_title=title)
    elseif num_slices == 1
        p = Plots.plot(size=(500, 500), right_margin = 30Plots.mm)
    end

    for slice_idx in 1:num_slices
        @info "Computing phase difference for slice $slice_idx"
        y = reconstruction[:, :, slice_idx, 1]
        z = reconstruction[:, :, slice_idx, 2]
        κ = conj.(y) .* z
        x = angle.(κ)
        # Compute pixel-wise phase difference (wrapped to [-π, π])
        #phase_diff = angle.(reconstruction[:, :, slice_idx, 2]) - angle.(reconstruction[:, :, slice_idx, 1])
        #phase_diff = mod.(phase_diff .+ π, 2π) .- π  # Wrap phase difference

        # Rotate image for proper orientation
        phase_diff = mapslices(x -> rotl90(x),x, dims=[1, 2])

        # Plot the image in the corresponding subplot
        Plots.heatmap!(p[slice_idx], phase_diff, color=:plasma, aspect_ratio=1, yflip=true, 
                       colorbar=true, grid=false, showaxis=false)
    end

    # Display the figure
    display(p)
end
"""
    plot_slices(reconstruction::Array{<:Complex, 4}, title::String; plot_phase::Bool=false)

Plot magnitude or phase images for each slice in a multi-slice image.

# Arguments
- `reconstruction`: Array of size `[Nx, Ny, num_slices, 1]`.
- `title`: Plot title.
- `plot_phase`: If `true`, plot wrapped phase; otherwise, normalized magnitude.

# Output
- Displays a grid of slice images.
"""

function plot_slices(reconstruction, title; plot_phase=false)
    num_slices = size(reconstruction, 3)  # Get the number of slices

    if num_slices == 15
        rows, cols = 3,5 # Define subplot grid
        # Create subplot layout
        p = Plots.plot(layout=(rows, cols), size=(2500, 1600), plot_title= title)
    elseif num_slices == 24
        rows, cols = 4,6
        p = Plots.plot(layout=(rows, cols), size=(2900, 1600), plot_title= title)
    elseif num_slices == 1
        p = Plots.plot(size=(300,300))
    end 
    

    for slice_idx in 1:num_slices
        if plot_phase
            # Extract and process phase image. Multiply with -1 to remove pi
            image_to_plot = angle.(reconstruction[:, :, slice_idx, 1].*(-1))
            colormap = :plasma
            #title_text = "Phase - Slice $slice_idx"
        else
            @info "Retrieving magnitude image for slice $slice_idx"
            # Extract and process magnitude image
            image_to_plot = mapslices(x -> abs.(x) ./ maximum(abs.(x)), reconstruction[:, :, slice_idx, 1], dims=[1, 2])
            image_to_plot = apply_windowing(image_to_plot, 3, 99.6)
            colormap = :grays
            #title_text = "Magnitude - Slice $slice_idx"
        end

        # Rotate image for proper orientation
        image_to_plot = mapslices(x -> rotl90(x), image_to_plot, dims=[1, 2])

        # Plot the image in the corresponding subplot
        Plots.heatmap!(p[slice_idx], image_to_plot, color=colormap, aspect_ratio=1, yflip=true, 
                       colorbar=false, grid=false, showaxis=false)
    end

    # Display the figure
    display(p)
end

"""
    plot_multiple_b0_maps(b0_map_volume; slice_indices, doSet_scale, doOffset, rad_s_offset)

Plots multiple B₀ field map slices in a grid layout.

# Arguments
- `b0_map_volume`: 3D array `[Nx, Ny, Nslices]` representing the B₀ field map (in rad/s).
- `slice_indices`: Optional array of slice indices to plot (default: all slices).
- `doSet_scale`: If `true`, uses a fixed color scale of ±2000 rad/s (default: `false`).
- `doOffset`: If `true`, adds a constant offset `rad_s_offset` to the map (default: `false`).
- `rad_s_offset`: Offset value in rad/s to add when `doOffset = true`.

# Output
Displays a grid of heatmaps using a consistent color scheme.
"""

function plot_multiple_b0_maps(b0_map_volume;
    slice_indices = 1:size(b0_map_volume, 3),
    doSet_scale = false,
    doOffset = false,
    rad_s_offset = 0)

    nslices = length(slice_indices)
    ncols = ceil(Int, sqrt(nslices))
    nrows = ceil(Int, nslices / ncols)

   

    # Apply offset if requested
    if doOffset
        b0_map_volume = b0_map_volume .+ rad_s_offset
    end

    p = Plots.plot(layout = (nrows, ncols), size = (ncols*350, nrows*350))#, right_margin = 30Plots.mm)

    for (i, slice_idx) in enumerate(slice_indices)
        b0_slice = b0_map_volume[:, :, slice_idx]
        b0_rotated = rotl90(b0_slice)
         # Use fixed color scale if requested
        if doSet_scale
            max_b0_value = 2000
        else
            max_b0_value = maximum(abs.(b0_map_volume[:,:,slice_idx]))
        end

        Plots.heatmap!(p[i], b0_rotated;
            color = :plasma,
            aspect_ratio = 1,
            yflip = true,
            colorrange = (-max_b0_value, max_b0_value),
            clims = (-max_b0_value, max_b0_value),
            colorbar = false,
            grid = false,
            showaxis = false)
            #title = "Slice $slice_idx")
    end

    display(p)
end


"""
    plot_reconstruction_simple(images, lower_perc, upper_perc; rotation=0, x_range=nothing, y_range=nothing)
    
Plots the magnitude and phase of the reconstructed images for a given slice or slices, WITHOUT B0 map.

# Arguments
* `images` - Complex-valued images reconstructed using MRIReco.jl
* `lower_perc, upper_perc` - Percentile values for intensity windowing
* `rotation::Int` - Counterclockwise rotation angle (must be 0, 90, 180, or 270)
* `x_range, y_range` (optional) - Pixel index ranges for the ROI box overlay
"""

function plot_reconstruction_simple(images, lower_perc, upper_perc; rotation=0, x_range=nothing, y_range=nothing)

    ## Ensure rotation is valid
    if mod(rotation, 90) != 0 || rotation < 0 || rotation > 270
        error("rotation must be 0, 90, 180, or 270 degrees.")
    end

    # Compute magnitude image and normalize
    image_to_plots = mapslices(x -> abs.(x) ./ maximum(abs.(x)), images[:, :, 1], dims=[1, 2])

    lower = percentile(vec(image_to_plots), lower_perc)
    upper = percentile(vec(image_to_plots), upper_perc)
    
    image_to_plots = clamp.((image_to_plots .- lower) ./ (upper - lower), 0, 1)
    # Compute percentiles for windowing
    

    # Apply windowing: clip and normalize
    #image_to_plots = clamp.((image_to_plots .- lower) ./ (upper - lower), 0, 1)

    # Apply rotation
    if rotation == 90
        image_to_plots = mapslices(x -> rotr90(x), image_to_plots, dims=[1, 2])
    elseif rotation == 180
        image_to_plots = mapslices(x -> rot180(x), image_to_plots, dims=[1, 2])
    else
        image_to_plots = mapslices(x -> rotl90(x), image_to_plots, dims=[1, 2])
    end

    # Create magnitude heatmap
    p1 = Plots.heatmap(image_to_plots, title="|Images|", color=:grays, aspect_ratio=1, yflip=true, colorbar=false, grid=false, showaxis=false)

    display(p1)

    # Compute phase image
    phase_images = angle.(images[:, :, 1, 1, 1].*(-1))

    # Apply rotation
    if rotation == 90
        phase_images = mapslices(x -> rotr90(x), phase_images, dims=[1, 2])
    elseif rotation == 180
        phase_images = mapslices(x -> rot180(x), phase_images, dims=[1, 2])
    else
        phase_images = mapslices(x -> rotl90(x), phase_images, dims=[1, 2])
    end

    # Create phase heatmap
    p2 = Plots.heatmap(phase_images, title="∠ Images (shifted by pi)", color=:plasma, aspect_ratio=1, yflip=true, colorbar=true, grid=false, showaxis=false)
    
    display(p2)
end


"""
    plot_kspace_trajectory_radm(non_corr_traj, girf_corr_traj, params_spiral, profile::Int)

Plots the 2D k-space trajectory before and after GIRF correction in physical units (rad/m).

# Arguments
- `non_corr_traj` :: Non-corrected trajectory object.
- `girf_corr_traj` :: GIRF-corrected trajectory object.
- `params_spiral` :: Dictionary containing gradient and trajectory parameters.
- `profile::Int` :: The profile index to plot.
"""
function plot_kspace_trajectory_radm(non_corr_traj, girf_corr_traj, params, profile::Int, index::Int)

    # Extract encoding parameters from params_spiral
    enc_fov = params[:enc_fov].*1e-3 # Field of view [m]
    enc_size = params[:enc_size]  # Encoding matrix size

    # Initialize the plot
    p = Plots.plot(title="2D k-space Trajectory (rad/m)", xlabel="k_x (rad/m)", ylabel="k_y (rad/m)", linewidth=2, aspect_ratio=1, legend=:topright)
    # Extract and plot the original trajectory
    kx_non_corr, ky_non_corr = extract_kspace(non_corr_traj[1],profile,enc_size, enc_fov)
    plot!(p, kx_non_corr[1:index], ky_non_corr[1:index], label="Original", linestyle=:solid)

    # Extract and plot the corrected trajectory
    kx_corr, ky_corr = extract_kspace(girf_corr_traj[1],profile,enc_size,enc_fov)
    plot!(p, kx_corr[1:index], ky_corr[1:index], label="GIRF corrected, 1st order", linestyle=:dash)

    # Display the plot
    display(p)

    savepath = params[:save_plot_path]

    Plots.savefig(p, savepath)
end

# Function to extract and convert k-space trajectory
function extract_kspace(trajectory,profile,enc_size,enc_fov)
    num_samples = trajectory.numSamplingPerProfile
    interleave_extractor = num_samples * (profile - 1) .+ (1:num_samples)

    # Extract k-space nodes
    k_x = trajectory.nodes[1, interleave_extractor]
    k_y = trajectory.nodes[2, interleave_extractor]

    # Convert to rad/m
    k_x_radm = k_x * (2 * π) * enc_size[1] / enc_fov[1]
    k_y_radm = k_y * (2 * π) * enc_size[2] / enc_fov[2]

    return k_x_radm, k_y_radm
end

###### Offset analysis #########

"""    
Plots the magnitude or phase images of spiral MRI reconstructions for different fieldmap offsets.

# Arguments:
* `params` - Dictionary containing general or spiral reconstruction parameters
* `fieldmap_offsets_Hz` - List of field-map offsets in Hz (integers)
* `plot_phase` - Boolean flag (default: `false`). If true, plots phase images instead of magnitude images.
"""
function load_and_plot_offset_recons(params, fieldmap_offsets_Hz; plot_phase=false)
    num_offsets = length(fieldmap_offsets_Hz)
    # Create subplot layout: 2 rows, N columns
    plot_grid = Plots.plot(layout = (1, num_offsets), size = (500 * num_offsets, 400))
    # Create a 3x3 subplot layout
    #p  = Plots.plot(layout=(3, 3), subplot_spacing=0.1, size=(1600, 1700)) # 1600,1700 for subplot of full images
    # Loop through each offset and plot the corresponding reconstruction
    x_range = 85:110 #81:105 #78:110
    y_range = 70:90
     

    for (i, offset) in enumerate(fieldmap_offsets_Hz)
        recon_path = joinpath(params_general[:recon_save_path], "current_spiral_$(slice_choice[1])_$(fm_offsets_Hz[i])Hz.nii")#joinpath(params[:recon_save_path],"_b0off_$(fm_offsets_Hz[i])Hz.nii")#joinpath(params[:recon_save_path], "$(offset)Hz.nii")
        @info "Loading reconstruction from: $recon_path"
        
        # Load the reconstructed image (complex-valued)
        recon = load_map(recon_path, do_split_phase=true)
        
        #recon = crop_image_by_mm(recon, 80.0, 180.0)

        if plot_phase
            # Extract and process phase image
            phase_image = angle.(recon[:, :, :, 1].*(-1))  # Extract phase
            phase_image = mapslices(x -> rotl90(x), phase_image, dims=[1, 2])  # Rotate
            image_to_plot = mosaicview(phase_image, nrow = Int(floor(sqrt(1))), npad = 5, rowmajor = true, fillvalue = 0)
            colormap = :plasma  # Colormap for phase
            title_text = "Phase: $(offset) Hz"
        else
            # Extract and process magnitude image
            image_to_plot = mapslices(x -> abs.(x) ./ maximum(abs.(x)), recon[x_range, y_range, :, 1], dims=[1, 2])
            image_to_plot = mapslices(x -> rotl90(x), image_to_plot, dims=[1, 2])
            image_to_plot = apply_windowing(image_to_plot, 3, 99.6) #crop_image_by_mm(60.0,30.0) for first slice first meas
            image_to_plot = mosaicview(image_to_plot, nrow = Int(floor(sqrt(1))), npad = 5, rowmajor = true, fillvalue = 0)
            colormap = :grays  # Colormap for magnitude
            #title_text = "Magnitude: $(offset) Hz"
        end

        # Plot the image in the corresponding subplot
        Plots.heatmap!(plot_grid[i], image_to_plot, color=colormap, aspect_ratio=:1, yflip=true, 
                       colorbar=false, grid=false, showaxis=false)
    end

    # Display the subplot figure
    display(plot_grid)
end

"""
    compare_reconstructions(cart_recon_up, fieldmap_offsets_Hz, params, x_range, y_range, lower_perc, upper_perc)

Compares a reference Cartesian reconstruction to a set of spiral reconstructions corrected with different field map offsets.

# Arguments
- `cart_recon_up`: Cartesian reference image `[Nx, Ny, 1, 1]`.
- `fieldmap_offsets_Hz`: Array of field map offsets (in Hz).
- `params`: Dictionary containing reconstruction paths.
- `x_range`, `y_range`: Pixel ranges for ROI cropping.
- `lower_perc`, `upper_perc`: Percentiles for intensity windowing.

# Returns
- Normalized RMSE values.
- Mutual information values.
- Normalized L1 distances.
- Normalized L2 norms.
"""
function compare_reconstructions(cart_recon_up, fieldmap_offsets_Hz, params, x_range, y_range, lower_perc, upper_perc)
    # Extract ROI from Cartesian reference
    roi_cart = cart_recon_up[x_range, y_range, 1, 1]
    roi_cart_mag = mapslices(x -> abs.(x) ./ maximum(abs.(x)), roi_cart, dims=[1, 2])
    roi_cart_mag = mapslices(rotl90, roi_cart_mag, dims=[1, 2])
    
    # Initialize metrics
    l1_values, rmse_values, l2_values, MI_values = Float64[], Float64[], Float64[], Float64[]
    num_offsets = length(fieldmap_offsets_Hz)
    spiral_images = Vector{Matrix{Float64}}(undef, num_offsets)
    diff_images = Vector{Matrix{Float64}}(undef, num_offsets)

    for (i, offset) in enumerate(fieldmap_offsets_Hz)
        # Load Spiral reconstruction
        recon_path = joinpath(params[:recon_save_path], "$(offset)Hz.nii")
        @info "Loading reconstruction from: $recon_path"
        recon_spiral = load_map(recon_path, do_split_phase=true)

        roi_spiral = recon_spiral[x_range, y_range, 1, 1]
        roi_spiral_mag = mapslices(x -> abs.(x) ./ maximum(abs.(x)), roi_spiral, dims=[1, 2])
        roi_spiral_mag = mapslices(rotl90, roi_spiral_mag, dims=[1, 2])

        # Metrics
        push!(l2_values, norm(roi_spiral_mag))
        diff_image = abs.(roi_cart_mag .- roi_spiral_mag) ./ maximum(roi_cart_mag)
        push!(l1_values, sum(diff_image))
        push!(rmse_values, sqrt(sum(diff_image .^ 2) / length(diff_image)))
        push!(MI_values, mutual_information(roi_cart_mag, roi_spiral_mag, 256))

        # Store windowed images
        spiral_images[i] = apply_windowing(roi_spiral_mag, lower_perc, upper_perc)
        diff_images[i] = apply_windowing(diff_image, lower_perc, upper_perc)
    end

    # Plotting
    plot_grid = Plots.plot(layout=(2, num_offsets), size=(500 * num_offsets, 800))
    for i in 1:num_offsets
        Plots.heatmap!(plot_grid[i], spiral_images[i], color=:grays, clims=(0, 1), yflip=true, aspect_ratio=1, showaxis=false)
        Plots.heatmap!(plot_grid[i + num_offsets], diff_images[i], color=:grays, clims=(-1, 1), yflip=true, aspect_ratio=1, showaxis=false)
    end
    display(plot_grid)

    return rmse_values ./ maximum(rmse_values), MI_values, l1_values ./ maximum(l1_values), l2_values ./ maximum(l2_values)
end


"""

mutual_information(img1, img2; nbins=256)

Computes the Mutual Information (MI) between two magnitude images.

# Arguments
- `img1`, `img2`: 2D arrays representing images (e.g., magnitude images from different reconstructions).
- `nbins`: Number of histogram bins used for the probability distribution (default: `256`).

# Returns
- `MI`: Scalar value representing the mutual information between `img1` and `img2`. A higher value indicates more shared information.

Uses normalized histograms to estimate marginal and joint entropies.
"""

function mutual_information(img1, img2, nbins=256)

    
    img1 = Float64.(vec(img1))
    img2 = Float64.(vec(img2))

    # Compute histograms
    hist_x = fit(Histogram, img1; nbins=nbins) # Histogram image 1
    hist_y = fit(Histogram, img2; nbins=nbins) # Histogram image 2

    # Normalize histograms to get probability distributions
    p_x = hist_x.weights ./ sum(hist_x.weights)
    p_y = hist_y.weights ./ sum(hist_y.weights) # .weights gives the counts of pixel intensities in each bin

    # Compute joint histogram (2D histogram)
    hist_xy = fit(Histogram, (img1, img2); nbins=(nbins, nbins))
    p_xy = hist_xy.weights ./ sum(hist_xy.weights)  # Normalize to probability distribution

    # Compute marginal and joint entropies, avoiding log(0)
    H_x = -sum(p_x .* log2.(p_x .+ eps()))  
    H_y = -sum(p_y .* log2.(p_y .+ eps()))
    H_xy = -sum(p_xy .* log2.(p_xy .+ eps()))

    # Compute Mutual Information (MI)
    return H_x + H_y - H_xy
end

