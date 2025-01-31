module SpaceTimeMultiscale

using NearestNeighbors
using LinearAlgebra
using Distances
using SparseArrays
using Random
using Distributions
using DataFrames, CSV
using PolyaGammaSamplers
using ProgressBars
using Optim
using Plots

using FiniteDiff


########################
# Structures
#########################

struct InputData{T}
    y::Vector{T}
    X::Matrix{Float64}
    loc::Matrix{Float64}
    time::Matrix{Float64}
end

export InputData

################

include("covariances.jl")

include("nngp.jl")

export getneighbors
export nngp
export nngp!
export nngppred
export nngppred!

include("misc.jl")

export quiltplot
export softmax
export getpropvar
export getlastsamp

include("priors.jl")
export gammaldens
export plotgamma

include("continuous_map.jl")
export continuous_map

include("samplers.jl")

include("continuous_mcmc.jl")
export continuous_mcmc

include("bernoulli_map.jl")
export bernoulli_map

include("bernoulli_mcmc.jl")
export bernoulli_mcmc
export bernoulli_hessian

include("continuous_loo.jl")
export continuous_loo

include("bernoulli_loo.jl")
export bernoulli_loo

include("bernoulli_lpmc.jl")
export bernoulli_lpmc
export bernoulli_lpmc_3o

end
