## File to plot dk/dt for the different spirals

#Our developed packages
using DelimitedFiles, Plots
using Plots, LaTeXStrings


data_root_project_path = "/Volumes/MasterB/ReconDirectory/Data/Gradients"

# Define spiral sequence names
spiral_names = ["140mm1p71mmR1", "140mm1p08mmR2", "192mm1p00mmR3", "140mm0p74mmR4"]
acceleration_factors = ["R = 1", "R = 2", "R = 3", "R = 4"]


# Define constants
#γ = 42577.478  # Gyromagnetic ratio in [Hz/mT]
dt_g = 1e-5   # Dwell time for gradient system
t_max_ms = 1
# Initialize plots
p1 = plot( xlabel="Time [ms]", ylabel="G_x [mT/m]")
p2 = plot(xlabel="Time [ms]", ylabel="G_y [mT/m]")

# Loop through each spiral sequence
for (i, spiral) in enumerate(spiral_names)
    gradient_file = joinpath(data_root_project_path,"ArbGradientRO_$spiral.txt")
    
    # Load gradient data
    G_data = readdlm(gradient_file)
    
    # Compute time axis in milliseconds
    t_in_ms = ((1:size(G_data, 1)) .* dt_g .- dt_g/2) .* 1e3  # Convert to ms

    # Extract only the first 2 ms of data
    idx_range = findall(t -> t <= t_max_ms, t_in_ms)
    t_short_ms = t_in_ms[idx_range]
    G_x_short = G_data[idx_range, 1]  # [mT/m]
    G_y_short = G_data[idx_range, 2]  # [mT/m]

    # Convert to γG_x and γG_y
    #γG_x = γ .* G_x_short
    #γG_y = γ .* G_y_short

    # Plot each gradient component for the first 2 ms
    plot!(p1, t_short_ms, G_x_short, label=acceleration_factors[i])
    plot!(p2, t_short_ms, G_y_short, label=acceleration_factors[i])
end

# Display plots
display(p1)
display(p2)
