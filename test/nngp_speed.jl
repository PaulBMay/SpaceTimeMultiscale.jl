using SpaceTimeMultiscale
using BenchmarkTools

n = 10*10^2
np = 100*10^2

loc = rand(n,2)
time = rand(n,1)

locpred = rand(np,2)
timepred = rand(np,1)

theta = exp.(rand(3,2))

nb = SpaceTimeMultiscale.getpredneighbors(loc, locpred, 25)

B, F, Border = SpaceTimeMultiscale.nngppred2(nb, loc, time, locpred, timepred, theta)

B2, F2, Border, NND = nngppred(nb, loc, time, locpred, timepred, theta)
