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


data = InputData(z, ones(n,1), loc, time)

##########################
###################

#plotgamma(0.05, 0.025^2, [0, 1])

thetapriors = (mu = copy(theta), var = [0.5 0.5; 0.05 0.1; 1 1].^2)
betapriors = (mu = [0], prec = [10])

##########

thetamap, logbook = bernoulli_map(theta, data, m, thetapriors, betapriors; f_tol = 1e-4, store_trace = true)

 thetamap = [2.07879    1.37599
            0.0827929  0.154579
            1.53123    1.7734
] 

Hess = bernoulli_hessian(thetamap, data, m, thetapriors, betapriors)

#Symmetric(logbook.trace[end].metadata["~inv(H)"])

outdir_lp = "./test/dump_lp/"

isdir(outdir_lp) || mkdir(outdir_lp)

bernoulli_lpmc(thetamap, data, m, Hess, betapriors, outdir_lp, 500)

outdir_lp3o = "./test/dump_lp3o/"

isdir(outdir_lp3o) || mkdir(outdir_lp3o)

bernoulli_lpmc_3o(thetamap, data, m, Hess, betapriors, outdir_lp3o, 500)


pardf = CSV.read(outdir_lp*"params.csv", DataFrame)

plot(pardf.sw1)
plot!(pardf.sw2)

plot(pardf.rangeS1)
plot!(pardf.rangeS2)

plot(pardf.rangeT1)
plot!(pardf.rangeT2)

####################

outdir_mc = "./test/dump_mc/"

isdir(outdir_mc) || mkdir(outdir_mc)

bernoulli_mcmc(theta, data, m, thetapriors, betapriors, outdir_mc, 500; adaptstart = 50, pgwarmup = 100)

lastsamp = reshape(getlastsamp(outdir_mc*"params.csv"), 3, :)

bernoulli_mcmc(lastsamp, data, m, thetapriors, betapriors, outdir_mc, 500; adaptstart = 50, pgwarmup = 100)


pardf_mc = CSV.read(outdir_mc*"params.csv", DataFrame)

plot(pardf_mc.sw1)
plot!(pardf_mc.sw2)

plot(pardf_mc.rangeS1)
plot!(pardf_mc.rangeS2)

plot(pardf_mc.rangeT1)
plot!(pardf_mc.rangeT2)

#rm(outdir, recursive = true)

######################


######################

lpd_lp = bernoulli_loo(outdir_lp, data)
lpd_lp3o = bernoulli_loo(outdir_lp3o, data)
lpd_mc = bernoulli_loo(outdir_mc,data)

histogram(lpd_mc, alpha = 0.5)
histogram!(lpd_lp3o, alpha = 0.5)

sum(lpd_lp)
sum(lpd_lp3o)
sum(lpd_mc)

scatter(lpd_lp3o, lpd_mc)
plot!([-3, 0], [-3, 0])


##

using SparseArrays


Dsgn = sparse_hcat(data.X, SpaceTimeMultiscale.speye(n))


effects_lp = CSV.read(outdir_lp*"effects.csv", Tables.matrix)
effects_lp3o = CSV.read(outdir_lp3o*"effects.csv", Tables.matrix)
effects_mc = CSV.read(outdir_mc*"effects.csv", Tables.matrix)


predsamps_lp = softmax.(Dsgn*effects_lp')
predsamps_lp3o = softmax.(Dsgn*effects_lp3o')
predsamps_mc = softmax.(Dsgn*effects_mc')


predmu_lp = mean(predsamps_lp, dims = 2)[:,1]
predmu_lp3o = mean(predsamps_lp3o, dims = 2)[:,1]
predmu_mc = mean(predsamps_mc, dims = 2)[:,1]


scatter(predmu_lp, predmu_mc, c = z)
plot!([0,1], [0,1])

scatter(predmu_lp3o, predmu_mc, c = z)
plot!([0,1], [0,1])

wtf = predmu2 .> predmu

quiltplot(loc, wtf)

scatter(predmu_lp, zprob)
scatter(predmu_lp3o, zprob)
scatter(predmu_mc, zprob)


######################

pos = data.y .== 1

sum(log.(predmu_lp[pos])) + sum(log.(1 .- predmu_lp[.!pos]))
sum(log.(predmu_lp3o[pos])) + sum(log.(1 .- predmu_lp3o[.!pos]))
sum(log.(predmu_mc[pos])) + sum(log.(1 .- predmu_mc[.!pos]))

rm(outdir_mc, recursive = true)
rm(outdir_lp, recursive = true)
rm(outdir_lp3o, recursive = true)

