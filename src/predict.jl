function continuous_predict(readdir::String, Xpred::AbstractArray, locpred::AbstractArray, timepred::AbstractArray, m::Integer)

    params = CSV.read(joinpath(readdir, "params.csv"), DataFrame)
    effects = CSV.read(joinpath(readdir, "effects.csv"), Tables.matrix)
    loctime = CSV.read(joinpath(readdir, "loctime.csv"), Tables.matrix)

    nsamps = size(params, 1)
    npred = size(locpred,1)
    n = size(loctime, 1)
    k = size(effects, 2)
    p = k - n

    beta = view(effects, :, 1:p)
    w = view(effects, :, (p+1):k)

    loc = view(loctime, :, 1:2)
    time = view(loctime, :, [3])

    nb = getpredneighbors(loc, locpred, m)

    theta = reshape(params[1,:], 3, :)

    B,F,Border = nngppred(nb, loc, time, locpred, timepred, theta)

    predsamps = zeros(nsamps, npred)

    @views for i in ProgressBar(1:nsamps)

        theta = reshape(params[i,:], 3, :)

        nngppred!(B, F, Border, nb, loc, time, locpred, timepred, theta)

        predsamps[i,:] = Xpred*beta[i,:] + B*w[i,:] + sqrt.(F).*randn(npred) + sqrt(params.tSq[i])*randn(npred)
    
    end

    return predsamps

end

function bernoulli_predict(readdir::String, Xpred::AbstractArray, locpred::AbstractArray, timepred::AbstractArray, m::Integer)

    params = CSV.read(joinpath(readdir, "params.csv"), DataFrame)
    effects = CSV.read(joinpath(readdir, "effects.csv"), Tables.matrix)
    loctime = CSV.read(joinpath(readdir, "loctime.csv"), Tables.matrix)

    nsamps = size(params, 1)
    npred = size(locpred,1)
    n = size(loctime, 1)
    k = size(effects, 2)
    p = k - n

    beta = view(effects, :, 1:p)
    w = view(effects, :, (p+1):k)

    loc = view(loctime, :, 1:2)
    time = view(loctime, :, [3])

    nb = getpredneighbors(loc, locpred, m)

    theta = reshape(params[1,:], 3, :)

    B,F,Border = nngppred(nb, loc, time, locpred, timepred, theta)

    predsamps = zeros(nsamps, npred)

    @views for i in ProgressBar(1:nsamps)

        theta = reshape(params[i,:], 3, :)

        nngppred!(B, F, Border, nb, loc, time, locpred, timepred, theta)

        predsamps[i,:] = softmax(Xpred*beta[i,:] + B*w[i,:] + sqrt.(F).*randn(npred))
    
    end

    return predsamps

end

function ratpwr(x, pwr)
    return  (abs(x)^pwr)*sign(x)
end

function agg_predict(readdirz::String, readdiry::String, Projection::AbstractArray, pwr::Number, Xpred::AbstractArray, locpred::AbstractArray, timepred::AbstractArray, m::Integer)

    paramsy = CSV.read(joinpath(readdiry, "params.csv"), DataFrame)
    effectsy = CSV.read(joinpath(readdiry, "effects.csv"), Tables.matrix)
    loctimey = CSV.read(joinpath(readdiry, "loctime.csv"), Tables.matrix)

    locy = view(loctimey, :, 1:2)
    timey = view(loctimey, :, [3])

    ny = size(loctimey, 1)
    ky = size(effectsy, 2)
    p = ky - ny

    betay = view(effectsy, :, 1:p)
    wy = view(effectsy, :, (p+1):ky)

    ############

    paramsz = CSV.read(joinpath(readdirz, "params.csv"), DataFrame)
    effectsz = CSV.read(joinpath(readdirz, "effects.csv"), Tables.matrix)
    loctimez = CSV.read(joinpath(readdirz, "loctime.csv"), Tables.matrix)

    locz = view(loctimez, :, 1:2)
    timez = view(loctimez, :, [3])

    nz = size(loctimez, 1)
    kz = size(effectsz, 2)

    betaz = view(effectsz, :, 1:p)
    wz = view(effectsz, :, (p+1):ky)

    #########

    nsamps = size(paramsy, 1)
    npred = size(locpred,1)
    nproj = size(Projection,2)

    #################
    
    nby = getpredneighbors(locy, locpred, m)
    nbz = getpredneighbors(locz, locpred, m)

    thetay = reshape(paramsy[1,:], 3, :)
    thetaz = reshape(paramsz[1,:], 3, :)

    By,Fy,Byorder = nngppred(nby, locy, timey, locpred, timepred, thetay)
    Bz,Fz,Bzorder = nngppred(nbz, locz, timez, locpred, timepred, thetaz)

    predsamps = zeros(nsamps, nproj)

    y = zeros(npred)
    zprob = zeros(npred)
    z = zeros(Integer, npred)

    @views for i in ProgressBar(1:nsamps)

        thetay = reshape(paramsy[i,:], 3, :)
        thetaz = reshape(paramsz[i,:], 3, :)

        nngppred!(By, Fy, Byorder, nby, locy, timey, locpred, timepred, thetay)
        nngppred!(Bz, Fz, Bzorder, nbz, locz, timez, locpred, timepred, thetaz)

        y .= Xpred*betay[i,:] + By*wy[i,:] + sqrt.(Fy).*randn(npred) + sqrt(paramsy.tSq[i])*randn(npred)
        zprob .= softmax(Xpred*betaz[i,:] + Bz*wz[i,:] + sqrt.(Fz).*randn(npred))
        z .= 1*(zprob .> rand(npred))

        predsamps[i,:] = Projection * (z .* y.^pwr)

    end

    return predsamps

end