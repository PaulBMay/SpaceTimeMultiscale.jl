using SpaceTimeMultiscale
using LinearAlgebra
using Random
using Plots
using CSV, DataFrames
using Distributions


ngridsqrt = 100
ngrid = ngridsqrt^2
ntime = 5

lonseq = range(0,1,ngridsqrt)
latseq = range(0,1,ngridsqrt)

grid = zeros(ngrid, 2)

for i in 1:ngridsqrt
    for j in 1:ngridsqrt
        row = (i-1)*ngridsqrt + j
        grid[row,:] = [lonseq[i] latseq[j]]
    end
end

locfull = repeat(grid, ntime)
timefull = 1.0*repeat(1:ntime, inner = ngrid)[:,:]
nfull = ngrid*ntime

####################

Random.seed!(96)

m = 25

thetay = [3 2; 0.05 0.2; 5 10]
tSq = 1^2
betay = 5

y = simulate_field(thetay, tSq, betay, locfull, timefull, 25)

#= t = 5
quiltplot(grid, y[(1:ngrid) .+ (t-1)*ngrid]) =#

#####

thetaz = [3 2; 0.05 0.2; 5 10]
betaz = 1

zprob = softmax.(simulate_field(thetaz, 0, betaz, locfull, timefull, 25))

z = 1*(zprob .> rand(nfull))

####

b = y .* z


n = 1000

trainind = shuffle(1:nfull)[1:n]

data = InputData(b[trainind], ones(n,1), locfull[trainind,:], timefull[trainind,:])

zdata, ydata = datasplit(data)

length(ydata.y)

##################

ythetapriors = (mu = copy(thetay), var = [0.5 0.5; 0.05 0.1; 1 1].^2)

tsqprior = (mu = 1, var = 1)

ythetamap, tSqmap, ylogbook = continuous_map(thetay, tSq, ydata, m, ythetapriors, tsqprior; f_tol = 1e-6, store_trace = true)

ythetamap
tSqmap

yHess = continuous_hessian(ythetamap, tSqmap, ydata, m, ythetapriors, tsqprior)

youtdir = "./test/contdump/"

isdir(youtdir) || mkdir(youtdir)

continuous_mcmc(ythetamap, tSqmap, ydata, m, ythetapriors, tsqprior, yHess, youtdir, 1000)


################

zthetapriors = (mu = copy(thetaz), var = [0.5 0.5; 0.05 0.1; 1 1].^2)
zbetapriors = (mu = [1], prec = [10])


############

zoutdir_mc = "./test/berndump_mc"

isdir(zoutdir_mc) || mkdir(zoutdir_mc)

propvar = 1e-4*Matrix(I,6,6)

bernoulli_mcmc(thetaz, propvar, zdata, m, zthetapriors, zbetapriors, zoutdir_mc, 1000; pgwarmup = 100)

lastsamp = reshape(getlastsamp(zoutdir_mc*"/params.csv"), 3, :)
propvar = getpropvar(zoutdir_mc*"/params.csv", 1000)
bernoulli_mcmc(lastsamp, propvar, zdata, m, zthetapriors, zbetapriors, zoutdir_mc, 2000; pgwarmup = 100)

###############

########################

nb = SpaceTimeMultiscale.getpredneighbors(ydata.loc, grid, 25)

rho, Ds, Dt, theta,i = nngppred(nb, ydata.loc, ydata.time, grid, fill(3.0, ngrid,1), thetay)

ypredsamps = continuous_predict(youtdir, ones(ngrid, 1), grid, fill(3.0, ngrid, 1), 25)
ypredmu = mean(ypredsamps, dims = 1)[1,:]

ytest = y[vec(timefull) .== 3.0]

scatter(ypredmu, ytest)
plot!([-20, 20], [-20, 20])

covered = zeros(Integer,ngrid)

for i in 1:ngrid

    quants = quantile(ypredsamps[:,i], [0.025, 0.975])

    if ((quants[1] < ytest[i]) && (ytest[i] < quants[2]))
        covered[i] = 1
    end

end

mean(covered)

######################

indpred = (locfull[:,1] .> 0.5) .&& (locfull[:,2] .> 0.5) .&& (timefull[:,1] .== 3.0)

locpred = locfull[indpred,:]
timepred = timefull[indpred,:]
Xpred = ones(sum(indpred), 1)
ytest = y[indpred]
zprobtest = zprob[indpred]
ztest = z[indpred]
btest = b[indpred]

zpredsamps = bernoulli_predict(zoutdir_lp, Xpred, locpred, timepred, m)

zpredsamps_mc = bernoulli_predict(zoutdir_mc, Xpred, locpred, timepred, m)

zpredmu = mean(zpredsamps, dims = 1)[1,:]
zpredmu_mc = mean(zpredsamps_mc, dims = 1)[1,:]

sum(ztest.*log.(zpredmu) + ( 1 .- ztest).*log.(1 .- zpredmu))
sum(ztest.*log.(zpredmu_mc) + ( 1 .- ztest).*log.(1 .- zpredmu_mc))


Projection = fill(1/sum(indpred), 1, sum(indpred))

aggsamps = agg_predict(zoutdir_mc, youtdir, Projection, 1, Xpred, locpred, timepred, m)


rm(zoutdir_lp, recursive = true)
rm(zoutdir_mc, recursive = true)
rm(youtdir, recursive = true)



