function bernoulli_newt!(mode::Vector, y::Vector, Dsgn::SparseMatrixCSC, Q::SparseMatrixCSC, Qpostchol::SparseArrays.CHOLMOD.Factor, betapriors::NamedTuple, tol::Real, maxiter::Integer)

    p = length(betapriors.mu)

    update = copy(mode)
    probs = softmax.(Dsgn*mode)
    Omega = spdiagm(probs.*(1 .- probs))
    Qpost = Q + Dsgn'*(Omega*Dsgn)
    cholesky!(Qpostchol, Hermitian(Qpost))
    grad = Dsgn'*(y - probs) - Q*mode
    grad[1:p] .+= betapriors.mu .* betapriors.prec
    mode .+= (Qpostchol \ grad)

    error = tol + 1.0
    count = 0

    while (error > tol) && (count <= maxiter)

        probs .= softmax.(Dsgn*mode)
        Omega .= spdiagm(probs.*(1 .- probs))
        Qpost .= Q + Dsgn'*(Omega*Dsgn)
        cholesky!(Qpostchol, Hermitian(Qpost))
        grad .= Dsgn'*(y - probs) - Q*mode
        grad[1:p] .+= betapriors.mu .* betapriors.prec
        update .= mode + (Qpostchol \ grad)
        error = norm(mode - update) / norm(update)
        mode .= copy(update)

        count += 1

    end

    return nothing


end

function bernoulli_lpmc(thetamap::AbstractArray, data::InputData, m::Integer, Hess::AbstractArray, betapriors::NamedTuple, outdir::String, nsamps::Integer; nr_tol = 1e-3, nr_maxiter = 30)

    ############################
    # Data dimensions and prep
    ##########################

    n = size(data.y, 1) # sample size
    p = size(data.X, 2) # num predictors
    ncomponents = size(thetamap, 2)

    # Does the out_dir exist?

    if !isdir(outdir)
        error("Can't find your outdir")
    end

    # Prepare CSV's

    loctimeout = joinpath(outdir, "loctime.csv")
    CSV.write(loctimeout, DataFrame(lon = data.loc[:,1], lat = data.loc[:,2], time = data.time[:,1]))

    paramsout = joinpath(outdir, "params.csv")
    paramsdf = DataFrame(zeros(1, ncomponents*3), repeat(["sw", "rangeS", "rangeT"], ncomponents) .* repeat(string.(1:ncomponents), inner = 3))

    effectsout = joinpath(outdir, "effects.csv")
    effectsdf = DataFrame(zeros(1,p+n), ["beta_".*string.(1:p); "w_".*string.(1:n)])

    # fixed/random effect values
    effects = zeros(p + n)
    mode = copy(effects)

    paramsdf[1,:] = vec(thetamap)
    effectsdf[1,:] = effects

    CSV.write(paramsout, paramsdf)
    CSV.write(effectsout, effectsdf)


    ####################
    # Lord have mercy that was boring.
    # Now fun stuff. Get the neighbor sets and initial NNGP mats
    #####################

    print("Getting neighbor sets\n")

    nb = getneighbors(data.loc, m)

    Dsgn = sparse_hcat(data.X, speye(n))

    print("Initial NNGP mats\n")

    theta = copy(thetamap)

    B,F,Border = nngp(nb, data.loc, data.time, theta)

    Q = blockdiag(spdiagm(betapriors.prec), (B'*spdiagm(1 ./ F)*B))

    Qpost = Q + Dsgn'*Dsgn

    Qpostchol = cholesky(Hermitian(Qpost))


    ###########

    lthetamap = log.(vec(thetamap))
    HessL = cholesky(Symmetric(Hess)).L

 

    for i in ProgressBar(1:nsamps)

        # Generate theta values

        ltheta = lthetamap + HessL*randn(ncomponents*3)
        theta .= exp.(reshape(ltheta, 3, ncomponents))
        
        nngp!(B, F, Border, nb, data.loc, data.time, theta)
        Q .= blockdiag(spdiagm(betapriors.prec), (B'*spdiagm(1 ./ F)*B))

        # Laplace simulation of effects

        bernoulli_newt!(mode, data.y, Dsgn, Q, Qpostchol, betapriors, nr_tol, nr_maxiter)

        effects .= mode + (Qpostchol.U \ randn(n+p))[invperm(Qpostchol.p)]

       #########################
       # Good work, team. Let's write out these results
       #########################

       paramsdf[1,:] = vec(theta)
       effectsdf[1,:] = effects

       CSV.write(paramsout, paramsdf; append = true, header = false)
       CSV.write(effectsout, effectsdf; append = true, header = false)


    end

    return nothing

end

function bernoulli_lpmc_3o(thetamap::AbstractArray, data::InputData, m::Integer, Hess::AbstractArray, betapriors::NamedTuple, outdir::String, nsamps::Integer; nr_tol = 1e-3, nr_maxiter = 30)

    ############################
    # Data dimensions and prep
    ##########################

    n = size(data.y, 1) # sample size
    p = size(data.X, 2) # num predictors
    ncomponents = size(thetamap, 2)

    # Does the out_dir exist?

    if !isdir(outdir)
        error("Can't find your outdir")
    end

    # Prepare CSV's

    loctimeout = joinpath(outdir, "loctime.csv")
    CSV.write(loctimeout, DataFrame(lon = data.loc[:,1], lat = data.loc[:,2], time = data.time[:,1]))

    paramsout = joinpath(outdir, "params.csv")
    paramsdf = DataFrame(zeros(1, ncomponents*3), repeat(["sw", "rangeS", "rangeT"], ncomponents) .* repeat(string.(1:ncomponents), inner = 3))

    effectsout = joinpath(outdir, "effects.csv")
    effectsdf = DataFrame(zeros(1,p+n), ["beta_".*string.(1:p); "w_".*string.(1:n)])

    # fixed/random effect values
    effects = zeros(p + n)
    mode = copy(effects)

    paramsdf[1,:] = vec(thetamap)
    effectsdf[1,:] = effects

    CSV.write(paramsout, paramsdf)
    CSV.write(effectsout, effectsdf)


    ####################
    # Lord have mercy that was boring.
    # Now fun stuff. Get the neighbor sets and initial NNGP mats
    #####################

    print("Getting neighbor sets\n")

    nb = getneighbors(data.loc, m)

    Dsgn = sparse_hcat(data.X, speye(n))

    print("Initial NNGP mats\n")

    theta = copy(thetamap)

    B,F,Border = nngp(nb, data.loc, data.time, theta)

    Q = blockdiag(spdiagm(betapriors.prec), (B'*spdiagm(1 ./ F)*B))

    Qpost = Q + Dsgn'*Dsgn

    Qpostchol = cholesky(Hermitian(Qpost))

    prob = softmax.(Dsgn*mode)
    w3 = @. prob*(1-prob)*(1 - 2*prob)
    v = Dsgn'*w3
    adj = Qpostchol \ v

    ###########

    lthetamap = log.(vec(thetamap))
    HessL = cholesky(Symmetric(Hess)).L

 

    for i in ProgressBar(1:nsamps)

        # Generate theta values

        ltheta = lthetamap + HessL*randn(ncomponents*3)
        theta .= exp.(reshape(ltheta, 3, ncomponents))
        
        nngp!(B, F, Border, nb, data.loc, data.time, theta)
        Q .= blockdiag(spdiagm(betapriors.prec), (B'*spdiagm(1 ./ F)*B))

        # Laplace simulation of effects

        bernoulli_newt!(mode, data.y, Dsgn, Q, Qpostchol, betapriors, nr_tol, nr_maxiter)

        prob .= softmax.(Dsgn*mode)
        w3 .= @. prob*(1-prob)*(1 - 2*prob)
        v .= Dsgn'*w3
        adj .= Qpostchol \ v

        effects .= mode - adj + (Qpostchol.U \ randn(n+p))[invperm(Qpostchol.p)]

       #########################
       # Good work, team. Let's write out these results
       #########################

       paramsdf[1,:] = vec(theta)
       effectsdf[1,:] = effects

       CSV.write(paramsout, paramsdf; append = true, header = false)
       CSV.write(effectsout, effectsdf; append = true, header = false)


    end

    return nothing

end

# Fix theta at the MAPs
function bernoulli_lpmc_3o(thetamap::AbstractArray, data::InputData, m::Integer, betapriors::NamedTuple, outdir::String, nsamps::Integer; nr_tol = 1e-3, nr_maxiter = 30)

    ############################
    # Data dimensions and prep
    ##########################

    n = size(data.y, 1) # sample size
    p = size(data.X, 2) # num predictors
    ncomponents = size(thetamap, 2)

    # Does the out_dir exist?

    if !isdir(outdir)
        error("Can't find your outdir")
    end

    # Prepare CSV's

    loctimeout = joinpath(outdir, "loctime.csv")
    CSV.write(loctimeout, DataFrame(lon = data.loc[:,1], lat = data.loc[:,2], time = data.time[:,1]))

    paramsout = joinpath(outdir, "params.csv")
    paramsdf = DataFrame(zeros(1, ncomponents*3), repeat(["sw", "rangeS", "rangeT"], ncomponents) .* repeat(string.(1:ncomponents), inner = 3))

    effectsout = joinpath(outdir, "effects.csv")
    effectsdf = DataFrame(zeros(1,p+n), ["beta_".*string.(1:p); "w_".*string.(1:n)])

    # fixed/random effect values
    effects = zeros(p + n)
    mode = copy(effects)

    paramsdf[1,:] = vec(thetamap)
    effectsdf[1,:] = effects

    CSV.write(paramsout, paramsdf)
    CSV.write(effectsout, effectsdf)


    ####################
    # Lord have mercy that was boring.
    # Now fun stuff. Get the neighbor sets and initial NNGP mats
    #####################

    print("Getting neighbor sets\n")

    nb = getneighbors(data.loc, m)

    Dsgn = sparse_hcat(data.X, speye(n))

    print("Initial NNGP mats\n")

    theta = copy(thetamap)

    B,F,Border = nngp(nb, data.loc, data.time, theta)

    Q = blockdiag(spdiagm(betapriors.prec), (B'*spdiagm(1 ./ F)*B))

    Qpost = Q + Dsgn'*Dsgn

    Qpostchol = cholesky(Hermitian(Qpost))

    bernoulli_newt!(mode, data.y, Dsgn, Q, Qpostchol, betapriors, nr_tol, nr_maxiter)

    prob = softmax.(Dsgn*mode)
    w3 = @. prob*(1-prob)*(1 - 2*prob)
    v = Dsgn'*w3
    adj = Qpostchol \ v

    mode .-= adj

    Qperm = invperm(Qpostchol.p)

    for i in ProgressBar(1:nsamps)

        effects .= mode + (Qpostchol.U \ randn(n+p))[Qperm]

        effectsdf[1,:] = effects

        CSV.write(effectsout, effectsdf; append = true, header = false)

    end


    return nothing

end