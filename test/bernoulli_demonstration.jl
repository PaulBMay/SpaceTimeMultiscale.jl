using SpaceTimeMultiscale
using LinearAlgebra
using Random
using Plots
using CSV, DataFrames

using BenchmarkTools

Random.seed!(96)

n = 2000
m = 25

theta = [3 2; 0.05 0.2; 1 2]

loc = rand(n,2)
time = rand(n,1)

nb = getneighbors(loc, m)


B, F, Border = nngp(nb, loc, time, theta)


u = randn(n)


simnngp =  ( LowerTriangular(B) \ (sqrt.(F) .* u) )

zprob = softmax.(2 .+ simnngp)

#quiltplot(loc, zprob)

z = 1 .* (zprob .> rand(n))

#quiltplot(loc, z)



#####################

data = InputData(z, ones(n,1), loc, time)

#plotgamma(0.05, 0.025^2, [0, 1])

thetapriors = (mu = copy(theta), var = [0.5 0.5; 0.05 0.1; 1 1].^2)
betapriors = (mu = [2], prec = [10])

#thetamap, logbook = bernoulli_map(theta, data, m, thetapriors, betapriors; f_tol = 1e-6, store_trace = true)

outdir = "./test/berndump/"

isdir(outdir) || mkdir(outdir)

bernoulli_mcmc(theta, data, m, thetapriors, betapriors, outdir, 10000; adaptstart = 50, pgwarmup = 100)

pardf = CSV.read("./test/berndump/params.csv", DataFrame)

plot(pardf.sw1)
plot!(pardf.sw2)

plot(pardf.rangeS1)
plot!(pardf.rangeS2)

plot(pardf.rangeT1)
plot!(pardf.rangeT2)

rm(outdir, recursive = true)
