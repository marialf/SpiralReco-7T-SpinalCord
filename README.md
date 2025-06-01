# SpiralReco-7T-SpinalCord
Codebase for spiral MRI reconstruction of spinal cord images at 7 Tesla, developed as part of my master’s thesis. The code builds on and depends on the open-source GIRFReco.jl framework.

Repository structure:

.
├── master_run_recon.jl           # Main script for single-slice reconstruction
├── multi_slice_run.jl           # Extension for multi-slice data
├── master_utils.jl              # Utility functions adapted from GIRFReco.jl
├── spiral_utils.jl              # Spiral-specific preprocessing utilities
├── CITATION.cff                 # Citation metadata
├── Project.toml / Manifest.toml # Julia environment files (optional)
└── README.md                    # You're here

Other code files are retrieved directly from GIRFReco.jl, as part of running spiral reconstruction with their packages. Please cite the original GIRFREco.jl publication if you use these components in your work:

Alexander Jaffray, Zhe Wu, S. Johanna Vannesjo, Kâmil Uludağ, Lars Kasper.
GIRFReco.jl: An Open-Source Pipeline for Spiral Magnetic Resonance Image (MRI) Reconstruction in Julia.
Journal of Open Source Software, 2024, 9(97), 5877.
https://doi.org/10.21105/joss.05877