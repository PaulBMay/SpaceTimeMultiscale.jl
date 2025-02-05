using SpaceTimeMultiscale

n = 20000

loc = rand(n,2)
time = rand(n,1)

theta = exp.(rand(3,2))

nb = SpaceTimeMultiscale.getneighbors(loc, 25)

B, F, Border = nngp(nb, loc, time, theta)

@time B,F,Border = nngp(nb, loc, time, theta)

@time B,F,Border = nngp2(nb, loc, time, theta)
