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

zprob = softmax.(0 .+ simnngp)

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

#= thetamap = [2.07879    1.37599
            0.0827929  0.154579
            1.53123    1.7734
] =#


Hess = bernoulli_hessian(thetamap, data, m, thetapriors, betapriors)

#Symmetric(logbook.trace[end].metadata["~inv(H)"])

outdir = "./test/dump_lp/"

isdir(outdir) || mkdir(outdir)

bernoulli_lpmc(thetamap, data, m, Hess, betapriors, outdir, 1000)

pardf = CSV.read(outdir*"params.csv", DataFrame)

plot(pardf.sw1)
plot!(pardf.sw2)

plot(pardf.rangeS1)
plot!(pardf.rangeS2)

plot(pardf.rangeT1)
plot!(pardf.rangeT2)

####################

outdir2 = "./test/dump_mc/"

isdir(outdir2) || mkdir(outdir2)

bernoulli_mcmc(theta, data, m, thetapriors, betapriors, outdir2, 500; adaptstart = 50, pgwarmup = 100)

lastsamp = reshape(getlastsamp(outdir2*"params.csv"), 3, :)

bernoulli_mcmc(lastsamp, data, m, thetapriors, betapriors, outdir2, 1000; adaptstart = 50, pgwarmup = 100)


pardf2 = CSV.read(outdir2*"params.csv", DataFrame)

plot(pardf2.sw1)
plot!(pardf2.sw2)

plot(pardf2.rangeS1)
plot!(pardf2.rangeS2)

plot(pardf2.rangeT1)
plot!(pardf2.rangeT2)

#rm(outdir, recursive = true)

######################

thetamap3 = reshape(mean(Matrix(pardf2), dims = 1), 3, 2) 

Hess3 = bernoulli_hessian(thetamap3, data, m, thetapriors, betapriors)

#Symmetric(logbook.trace[end].metadata["~inv(H)"])

outdir3 = "./test/dump_lp3/"

isdir(outdir3) || mkdir(outdir3)

bernoulli_lpmc(thetamap3, data, m, Hess3, betapriors, outdir3, 5000)

pardf3 = CSV.read("./test/dump_lp3/params.csv", DataFrame)

plot(pardf3.sw1)
plot!(pardf3.sw2)

plot(pardf3.rangeS1)
plot!(pardf3.rangeS2)

plot(pardf3.rangeT1)
plot!(pardf3.rangeT2)


######################

lpd = bernoulli_loo(outdir, data)
lpd2 = bernoulli_loo(outdir2, data)
lpd3 = bernoulli_loo(outdir3,data)

histogram(lpd, alpha = 0.5)
histogram!(lpd2, alpha = 0.5)

sum(lpd)
sum(lpd2)
sum(lpd3)

scatter(lpd, lpd2)
plot!([-3, 0], [-3, 0])
scatter(lpd3, lpd2)


##

using SparseArrays


Dsgn = sparse_hcat(data.X, SpaceTimeMultiscale.speye(n))


effects = CSV.read(outdir*"effects.csv", Tables.matrix)
effects2 = CSV.read(outdir2*"effects.csv", Tables.matrix)
effects3 = CSV.read(outdir3*"effects.csv", Tables.matrix)


predsamps = softmax.(Dsgn*effects')
predsamps2 = softmax.(Dsgn*effects2')
predsamps3 = softmax.(Dsgn*effects3')


predmu = mean(predsamps, dims = 2)[:,1]
predmu2 = mean(predsamps2, dims = 2)[:,1]
predmu3 = mean(predsamps3, dims = 2)[:,1]


scatter(predmu, predmu2, c = z)
plot!([0,1], [0,1])

scatter(predmu3, predmu)
plot!([0,1], [0,1])

wtf = predmu2 .> predmu

quiltplot(loc, wtf)

scatter(predmu, zprob)
scatter(predmu2, zprob)
scatter(predmu3, zprob)


######################

pos = data.y .== 1

sum(log.(predmu[pos])) + sum(log.(1 .- predmu[.!pos]))
sum(log.(predmu2[pos])) + sum(log.(1 .- predmu2[.!pos]))
sum(log.(predmu3[pos])) + sum(log.(1 .- predmu3[.!pos]))

rm(outdir, recursive = true)
rm(outdir2, recursive = true)
rm(outdir3, recursive = true)

