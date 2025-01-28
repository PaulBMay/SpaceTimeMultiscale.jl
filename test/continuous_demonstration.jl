using SpaceTimeMultiscale
using LinearAlgebra
using Random
using Plots
using CSV, DataFrames

using BenchmarkTools

Random.seed!(96)

n = 1000
m = 25

theta = [3 2; 0.05 0.2; 1 2]
tSq = 0.1^2

loc = rand(n,2)
time = rand(n,1)

nb = getneighbors(loc, m)


B, F, Border = nngp(nb, loc, time, theta)

u = randn(n)

w =  ( LowerTriangular(B) \ (sqrt.(F) .* u) )

#quiltplot(loc, w)


y = 2 .+ w + sqrt(tSq)*randn(n)


#####################

data = InputData(y, ones(n,1), loc, time)

#plotgamma(1, 1, [0, 1])

thetapriors = (mu = copy(theta), var = [0.5 0.5; 0.05 0.1; 1 1].^2)

tsqprior = (mu = 1, var = 1)

thetamap, tSqmap, logbook = continuous_map(theta, tSq, data, m, thetapriors, tsqprior; f_tol = 1e-6, store_trace = true)

thetamap
tSqmap

Hess = Matrix(Symmetric(logbook.trace[end].metadata["~inv(H)"]))

outdir = "./test/contdump/"

isdir(outdir) || mkdir(outdir)

continuous_mcmc(theta, tSq, data, m, thetapriors, tsqprior, Hess, outdir, 10000)

pardf = CSV.read("./test/contdump/params.csv", DataFrame)

plot(pardf.sw1)
plot!(pardf.sw2)

plot(pardf.rangeS1)
plot!(pardf.rangeS2)

plot(pardf.rangeT1)
plot!(pardf.rangeT2)

################
# Compare to single component

thetapriors2 = (mu = [3.5; 0.1; 1.5][:,:], var = [1 ; 0.05; 1][:,:].^2)

thetamap2, tSqmap2, logbook2 = continuous_map(thetapriors2.mu, tSq, data, m, thetapriors2, tsqprior; f_tol = 1e-6, store_trace = true)

thetamap2

Hess2 = Matrix(Symmetric(logbook2.trace[end].metadata["~inv(H)"]))

outdir2 = "./test/contdump2/"

isdir(outdir2) || mkdir(outdir2)

continuous_mcmc(thetamap2, tSqmap2, data, m, thetapriors2, tsqprior, Hess2, outdir2, 10000)


pardf2 = CSV.read("./test/contdump2/params.csv", DataFrame)

plot(pardf2.sw1)
#plot!(pardf2.sw2)

plot(pardf2.rangeS1)
#plot!(pardf2.rangeS2)

plot(pardf2.rangeT1)
#plot!(pardf2.rangeT2)

#####################
# LOO stats


lpd = continuous_loo(outdir, data)

histogram(lpd, alpha = 0.2)

lpd2 = continuous_loo(outdir2, data)

sum(lpd)
sum(lpd2)

histogram!(lpd2, alpha = 0.2)




fx  = CSV.read(outdir*"effects.csv", Tables.matrix)
fx2  = CSV.read(outdir2*"effects.csv", Tables.matrix)

using SparseArrays

Dsgn = sparse_hcat(data.X, SpaceTimeMultiscale.speye(n))

fit = Dsgn*fx'
fit2 = Dsgn*fx2'

fitmu = mean(fit, dims = 2)[:,1]
fitmu2 = mean(fit2, dims = 2)[:,1]


scatter(fitmu, data.y)

scatter(fitmu2, data.y)

###############
###############

rm(outdir, recursive = true)
rm(outdir2, recursive = true)
