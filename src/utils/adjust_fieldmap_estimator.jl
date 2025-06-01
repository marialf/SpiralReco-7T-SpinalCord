#------------------------------------------------------------------------------------
#  [SpiralReco-7T-SpinalCord](@id adjust_fieldmap_estimator.jl)
#------------------------------------------------------------------------------------

#
#  File: adjust_fieldmap_estimator.jl
#  Project: Master's Thesis – B0-corrected single-shot spiral MRI of the cervical spinal cord at 7 Tesla
#  Author: [Maria Leseth Føyen], NTNU, Department of Physics
#  Supervisor: [Johanna Vannesjö]
#  Date: [June 2025]
#
#  Description:
# Adaptation of the GIRFReco.jl fieldmap_estimator.jl, 
#to add a constant phase shift of chosen value to the complex phase image before field map estimation, with goal to mitigate phase wrapping
#
#  This code builds on and extends the open-source GIRFReco.jl framework.
#  Repository: https://github.com/YourUsername/SpiralReco-7T-SpinalCord
#
#-----------------------------------------------------------------------------------

using Flux



"""
ml_cost(x::Matrix{T},y::Matrix{Complex{T}},z::Matrix{Complex{T}}, β)
Calculates the ML estimator cost between an estimated phase map and the underlying multi-echo scan data
# Arguments
* `x::Matrix{T}` - fieldmap estimate (in radians)
* `y::Matrix{Complex{T}}` - Complex first-echo image data
* `z::Matrix{Complex{T}}` - Complex second-echo image data
* `m::Matrix{T}` - normalized weighting data (m ∈ [0,1]:= abs.(conj.(y).*z)./maximum(abs.(conj.(y).*z))), precomputed for speed
* `β` - Regularization parameter controlling roughness penalty
"""
function my_ml_cost(x::Matrix{T}, y::Matrix{Complex{T}}, z::Matrix{Complex{T}}, m::Matrix{T}, β) where {T}

    Ψ = (sum(m .* (1 .- cos.(angle.(z) .- angle.(y) .- x)), dims = [1, 2])+β*R_func(x))[1]

end

"""
R(x::Matrix{T})
Regularization function which penalizes roughness
# Arguments
* `x::Matrix{T}` - fieldmap estimate (in radians)
"""
function R_func(x::Matrix{T}) where {T}

    return T.(0.5) * sum(abs2, diff(x, dims = 1), dims = [1, 2]) + T.(0.5) * sum(abs2, diff(x, dims = 2), dims = [1, 2])

end

"""
pcg_ml_est_fieldmap(y::AbstractMatrix{Complex{T}},z::AbstractMatrix{Complex{T}},β) 
Estimates the fieldmap using the method presented in https://doi.org/10.1109/tmi.2008.923956
# Required Arguments
* `y::AbstractMatrix{Complex{T}}` - Complex first-echo image data
* `z::AbstractMatrix{Complex{T}}` - Complex second-echo image data

# Optional Arguments
* `β` - Regularization parameter controlling roughness penalty
* `reltol` - early stopping criteria (exit if subsequent cost function change < reltol)
"""
function my_pcg_ml_est_fieldmap(y::AbstractMatrix{Complex{T}}, z::AbstractMatrix{Complex{T}}, β = 1e-3, reltol = 5e-3) where {T}

    d = 1
    κ = conj.(y) .* z
    shift_value = 1.6

    k_shifted =  κ .* exp(-im * shift_value)  # Apply phase shift
    x_shifted = Float32.(angle.(k_shifted))  # Phase map after shift
    # Shift x back, where x_shift != 0
    mask = abs.(x_shifted) .!= 0.0
    x = Float32.(x_shifted .+ mask .* shift_value)
    @assert size(x) == size(y) "Mismatch in dimensions: x must be same size as y"

    m = Float32.(abs.(κ) ./ maximum(abs.(κ)))

    c = 0
    Δ = 1
    itcount = 0

    trust_step = 1.0 ./ (m .+ 4 * β)

    while Δ > reltol

        gs = gradient(Flux.params(x)) do
            c = my_ml_cost(x, y, z, m, β) # need to interpolate the y z and β to have better performance 
        end

        x̄ = gs[x]
        x .-= (trust_step) .* x̄

        Δ = abs.(my_ml_cost(x, y, z, m, β) - c) / c

        itcount += 1

    end

    print("Required $itcount iterations to converge below tolerance $reltol ")

    return x

end

"""
estimate_b0_maps(im_data,slices, TE1,TE2,β,isrotated)
Processes 3D volume data as output from MRIReco.reconstruction to estimate fieldmaps using the method presented by Funai and Fessler
# Required Arguments
* `im_data` - 5-D array with coEto process (must be within range of 3rd dimension of im_data)
* `TE1` - Echo time 1 [ms]
* `TE2` - Echo time 2 [ms]

# Optional Arguments
* `isrotated` - Boolean controlling whether to rotate the B0 maps to match the images or not (legacy feature)

# Keyword Arguments 
* `β` - Regularization parameter controlling roughness penalty (larger = smoother)
* `reltol` - early stopping criteria (exit if subsequent cost function change < reltol)

"""
function my_estimate_b0_maps(im_data, slices, TE1, TE2, isrotated::Bool = false; β = 5e-4, reltol = 0.001)

    b0_maps = Complex.(zeros(size(im_data)[1:3]))

    @sync for slice in slices
        Threads.@spawn begin
            # Maria modified: tried multiplying with exp(-i pi)
            b0_maps[:, :, slice] = my_pcg_ml_est_fieldmap(im_data[:, :, slice, 1, 1], im_data[:, :, slice, 2, 1], β, reltol) ./ ((TE2 - TE1) / 1000)
            println("for slice $slice")
        end
    end

    b0_maps = mapslices(isrotated ? x -> x : x -> rotl90(x), b0_maps, dims = (1, 2))

    return real.(b0_maps)

end
