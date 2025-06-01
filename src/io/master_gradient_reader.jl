#------------------------------------------------------------------------------------
#  [SpiralReco-7T-SpinalCord](@id master_gradient_reader.jl)
#------------------------------------------------------------------------------------

#
#  File: adjust_fieldmap_estimator.jl
#  Project: Master's Thesis – B0-corrected single-shot spiral MRI of the cervical spinal cord at 7 Tesla
#  Author: [Maria Leseth Føyen], NTNU, Department of Physics
#  Supervisor: [Johanna Vannesjö]
#  Date: [June 2025]
#
#  Description:
# Adaptation of the GIRFReco.jl fgradient_reader.jl, 
#   for customized read in of gradient waveform data, from nominal gradient .txt file. 
#
#  This code builds on and extends the open-source GIRFReco.jl framework.
#  Repository: https://github.com/YourUsername/SpiralReco-7T-SpinalCord
#
#-----------------------------------------------------------------------------------

"""
# Arguments
* `filename` - filename (with full path) of text file with gradient waveform information, 4000 sampling points (can change?)
* `fov:: Tuple{Int64,Int64,Int64}` - size of reconstructed image (trailing dimension 1 for 2D acquisitions)
        from params[:fov] = [192,192,1] [mm]

function read_gradient_text_file(filename, reconsize, delay)
"""


function my_read_gradient_text_file(filename, reconsize, enc_fov, enc_size)

    G_data = readdlm(filename)
    G_length = size(G_data, 1)

    dt_g = 1e-5 # Dwell time for gradient system

    # Time axis of gradient
    t_in = (1:size(G_data, 1)) .* dt_g # Time axis for G 

    # Zero padding
    ext = Int(size_res - G_length)
    G_ext = vcat(G_data, zeros(ext, 3))  #[mT/m] original
    t_in = vcat(t_in, t_in[end] .+ (1:ext) .* dt_g)  # Extend t_nom accordingly

    ### Initialize gradient vector, for now only necessary parameters for recon without GIRF 

    # Parameters for the dictionary
    n_samples = length(t_in)
    # Keep the same as recon_size for now
    # Reconstruction fov?
    fov_m = enc_fov.* 1e-3 #[m]
    voxelsize = fov_m ./reconsize
    acq_duration = n_samples * dt_g # s

    gradient_dict = Dict{Symbol,Any}()
    #gradient_dict[:version_number] = "#4"
    gradient_dict[:dwell_time] = dt_g # [seconds], dwell time gradient system 
    gradient_dict[:delay] = 0.0
    gradient_dict[:samples_per_interleave] =  n_samples 
    gradient_dict[:num_interleaves] = 1
    gradient_dict[:num_dims] = 2 # G_x and G_y
    gradient_dict[:time_to_center_kspace] = 0.0 # [seconds]
    gradient_dict[:acq_duration] = acq_duration #[seconds]
    gradient_dict[:echo_time_shift_samples] = 0.0
    gradient_dict[:enc_fov] = fov_m # [m], 
    gradient_dict[:voxel_dims] = voxelsize
    gradient_dict[:gradient_strength_factor] =  maximum(abs.(G_ext)) # [mT/m] Not necessary to multiply with here, never normalized with this in the first place?
    gradient_dict[:is_binary] = 0
    gradient_dict[:gamma] = 42577.478 # [Hz/mT] 
    gradient_dict[:field_strength] = 7 # [T] CAN CHANGE

   println("Gradient strength factor is: ", gradient_dict[:gradient_strength_factor])
    # Fill in the gradient vector

    # Extract G_x and G_y gradients 
    G_x = G_ext[:, 1] #[mT/m]
    G_y = G_ext[:, 2] #[mT/m]

    # Combine G_x and G_y into a 2D array of shape (size_res, 2)
    gradient_2d = hcat(G_x, G_y)  # Create a 2D array with G_x and G_y as columns

    gradient_array = reshape(gradient_2d, size_res, gradient_dict[:num_interleaves] , gradient_dict[:num_dims])

    gradient_dict[:gradient_vector] = gradient_array

    gradient_array_new = Array{Float64,3}(undef, size(gradient_array))

    # k_times in seconds, dwell_time/2 compensates for integration, such that it starts at time 0. 
    k_times = t_in .- gradient_dict[:dwell_time]./2 .+ gradient_dict[:delay] # Delay = 0 for now.

    ## Loop over all of the unique excitation trajectories and create an interpolant of the gradient
    for dim = 1:gradient_dict[:num_dims]

        for l = 1:gradient_dict[:num_interleaves]

            #print((dim,l),"\n")

            sp = Spline1D(planned_times, gradient_array[:, l, dim], w = ones(length(planned_times)), k = 1, bc = "zero", s = 0.0)

            # evaluate the interpolant at the sampling times of the kspace data
            gradient_array_new[:, l, dim] = sp(k_times)

            print(gradient_array_new[:,l,dim][end],"\n")

        end

    end

    ## cumulative summation and numerical integration of the gradient data, resulting in the kspace trajectory
    kspace_trajectory_array_new = gradient_dict[:gamma] * gradient_dict[:dwell_time] * cumsum(gradient_array_new, dims = 1) # [rad/m]

    converted_kspace_trajectory_array_new = kspace_trajectory_array_new 
    converted_kspace_trajectory_array_new[:, :, 1] *=  gradient_dict[:enc_fov][1]./ enc_size[1] 
    converted_kspace_trajectory_array_new[:, :, 2] *=  gradient_dict[:enc_fov][2]./ enc_size[2]

    ## Reshaping of the array to the format expected by the Trajectory constructor in MRIReco.jl
    # - dim 1 = kspace dimension
    # - dim 2 = kspace position (with interleaves/profiles arranged consecutively)
    permuted_trajectory =
        permutedims(reshape(converted_kspace_trajectory_array_new, gradient_dict[:samples_per_interleave] * gradient_dict[:num_interleaves], gradient_dict[:num_dims]), [2, 1])

    trajectory_object = Trajectory(
        permuted_trajectory,
        1, # Endret til 20 for å se om det funker mtp å rekonstruere alle profiles
        gradient_dict[:samples_per_interleave],
        TE = gradient_dict[:echo_time_shift_samples],
        AQ = gradient_dict[:acq_duration],
        numSlices = 15,
        cartesian = false,
        circular = true,
    )

    return trajectory_object

end



