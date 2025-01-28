function continuous_theta_lp(theta::AbstractArray, tSq::Real, Qpostchol::SparseArrays.CHOLMOD.Factor, F::Vector, betaprec::Vector, Dsgn::SparseMatrixCSC, y::Vector, thetapriors::NamedTuple, tSqprior::NamedTuple)


    n = length(y)
    ncomponents = size(theta,2)
    
    local ysolve = (y ./ tSq) - (Dsgn*(Qpostchol \ (Dsgn'*y) ) ./ tSq^2)

    local sse = dot(y, ysolve)

    local priorldet = sum(log.(betaprec)) - sum(log.(F))

    local ldet = n*log(tSq) + logdet(Qpostchol) - priorldet

    local ll = -0.5*(sse + ldet)

    lprior = 0.0
    for i in 1:3
        for j in 1:ncomponents
            lprior += gammaldens(theta[i,j], thetapriors.mu[i,j], thetapriors.var[i,j])
        end
    end

    lprior += gammaldens(tSq, tSqprior.mu, tSqprior.var)

    local lp = ll + lprior

    return lp


end


function continuous_mcmc(theta::AbstractArray, tSq::Real, data::InputData, m::Integer, thetapriors::NamedTuple, tSqprior::NamedTuple, parvar::AbstractArray, outdir::String, nsamps::Int64; fixpars = false)

    ############################
    # Data dimensions and prep
    ##########################

    n = size(data.y, 1) # sample size
    p = size(data.X, 2) # num predictors
    ncomponents = size(theta, 2) # num separable components
    npars = 3*ncomponents + 1 # number of hyperparameters


    ##########################
    # Initial values and CSV creation
    ##########################

    # Does the out_dir exist?

    if !isdir(outdir)
        error("Can't find your outdir")
    end

    # Prepare CSV's

    loctimeout = joinpath(outdir, "loctime.csv")
    CSV.write(loctimeout, DataFrame(lon = data.loc[:,1], lat = data.loc[:,2], time = data.time[:,1]))

    paramsout = joinpath(outdir, "params.csv")
    paramsdf = DataFrame(zeros(1, npars), 
        [repeat(["sw", "rangeS", "rangeT"], ncomponents) .* repeat(string.(1:ncomponents), inner = 3);  
        "tSq"]
        )

    effectsout = joinpath(outdir, "effects.csv")
    effectsdf = DataFrame(zeros(1,p+n), ["beta_".*string.(1:p); "w_".*string.(1:n)])
    

    # Parameter/effect values

    effects = zeros(p + n)
    beta = view(effects, 1:p)
    w = view(effects, (p+1):(p+n))

    paramsdf[1,:] = [vec(theta); tSq]
    effectsdf[1,:] = effects

    CSV.write(paramsout, paramsdf)
    CSV.write(effectsout, effectsdf)


    ####################
    #Get the neighbor sets and initial NNGP mats
    #####################


    println("Getting neighbor sets")

    nb = getneighbors(data.loc, m)

    println("Initial NNGP mats")

    B,F,Border = nngp(nb, data.loc, data.time, theta)

    Dsgn = sparse_hcat(data.X, speye(n)) 

    yproj = Dsgn'*(data.y ./ tSq)

    betaprec = fill(0.01, p)

    ##############

    Qpost = blockdiag(
        spdiagm(betaprec),
        B'*spdiagm(1 ./ F)*B
    ) + (1/tSq)*Dsgn'*Dsgn

    Qpostchol = cholesky(Hermitian(Qpost))
    Qpostcholprop = copy(Qpostchol)

    currentpars = log.([vec(theta); tSq])
    proppars = copy(currentpars)
    logpost = continuous_theta_lp(theta, tSq, Qpostchol, F, betaprec, Dsgn, data.y, thetapriors, tSqprior)
    logpostprop = logpost
    acceptpars = false

    propchol = cholesky(parvar).L



    #########################
    # Begin Gibbs sampler
    #########################

    for i = ProgressBar(1:nsamps)

       ########################
       # Sample effects (beta, w)
       ########################

       effects .= getgausssamp(Qpostchol, yproj)

       ###########################
       # Sample all spatial parameters associated with y
       ###########################

       if !fixpars

        proppars = currentpars + propchol*randn(npars)

        thetaprop = exp.( reshape(proppars[1:(end-1)], 3, ncomponents) )
        tSqprop = exp(proppars[end])

        # Get NNGP matrices associated with the proposal values
        nngp!(B, F, Border, nb, data.loc, data.time, thetaprop)

        Qpost .= blockdiag(
                spdiagm(betaprec),
                B'*spdiagm(1 ./ F)*B
        ) + (1/tSqprop)*Dsgn'*Dsgn

        cholesky!(Qpostcholprop, Hermitian(Qpost))

        logpostprop = continuous_theta_lp(thetaprop, tSqprop, Qpostcholprop, F, betaprec, Dsgn, data.y, thetapriors, tSqprior)

        acceptprob = exp.(logpostprop + sum(proppars)  - logpost - sum(currentpars))

        acceptpars = rand(1)[1] < acceptprob

        if acceptpars

            theta, tSq = copy(thetaprop), tSqprop
            Qpostchol = copy(Qpostcholprop)
            logpost = logpostprop
            currentpars .= copy(proppars)
            yproj .= Dsgn'*(data.y ./ tSq)

        end

       end


       #########################
       # Good work, team. Let's write out these results
       #########################

       effectsdf[1,:] = effects
       CSV.write(effectsout, effectsdf; append = true, header = false)

       if !fixpars
        paramsdf[1,:] = [vec(theta); tSq]
        CSV.write(paramsout, paramsdf; append = true, header = false)
       end

    end



    return nothing


end