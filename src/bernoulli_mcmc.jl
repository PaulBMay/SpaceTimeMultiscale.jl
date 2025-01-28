# adaptive proposal variance for theta
function bernoulli_mcmc(theta::AbstractArray, data::InputData, m::Int64, thetapriors::NamedTuple, betapriors::NamedTuple, outdir::String, nsamps::Int64; adaptstart = 50, pgwarmup = 100)


    ############################
    # Data dimensions and prep
    ##########################

    n = size(data.y, 1) # sample size
    p = size(data.X, 2) # num predictors
    ncomponents = size(theta, 2)

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
    beta = view(effects, 1:p)
    w = view(effects, (p+1):(p+n))

    paramsdf[1,:] = vec(theta)
    effectsdf[1,:] = effects

    CSV.write(paramsout, paramsdf)
    CSV.write(effectsout, effectsdf)


    ####################
    # Lord have mercy that was boring.
    # Now fun stuff. Get the neighbor sets and initial NNGP mats
    #####################

    print("Getting neighbor sets\n")

    nb = getneighbors(data.loc, m)

    print("Initial NNGP mats\n")

    B,F,Border = nngp(nb, data.loc, data.time, theta)
    Bprop,Fprop = copy(B), copy(F)

    Dsgn = sparse_hcat(data.X, speye(n))

    zproj = Dsgn'*(data.y .- 0.5)

    zproj[1:p] += betapriors.mu .* betapriors.prec

    pg = rpg.(fill(0.3, n))




    ##############

    Q = blockdiag(
        spdiagm(betapriors.prec),
        B'*spdiagm(1 ./ F)*B
    )

    Qpost = Q + Dsgn'*spdiagm(pg)*Dsgn

    Qpostchol = cholesky(Hermitian(Qpost))

    accepttheta = 0

    lthetamat = zeros(nsamps+1, ncomponents*3)
    lthetamat[1,:] = log.(vec(theta))

    lthetavar = 1e-5*Matrix(I,3*ncomponents,3*ncomponents)

    #####################
    # pg warmup
    #####################

    println("Warming up Polya-Gamma and random effect values")

    for i = 1:pgwarmup

       Qpost .= Q + Dsgn'*spdiagm(pg)*Dsgn

       effects .= getgausssamp!(Qpostchol, Qpost, zproj)

       pg .= rpg.(Dsgn*effects)

    end

    Q = []


    #########################
    # Begin Gibbs sampler
    #########################

    for i = ProgressBar(1:nsamps)

       ############################
       # Sample beta, w
       ############################

       Qpost .= blockdiag(
            spdiagm(betapriors.prec),
            B'*spdiagm(1 ./ F)*B
        ) + Dsgn'*spdiagm(pg)*Dsgn

       effects .= getgausssamp!(Qpostchol, Qpost, zproj)

       #######################
       # Sample pg
       #######################

       pg .= rpg.(Dsgn*effects)

       ###########################
       # Sample sw, rangeS, rangeT
       ###########################

       if i >= adaptstart
        lthetavar .= (2.4^2/(3*ncomponents))*cov(lthetamat[1:i,:])
       end

       currentltheta = lthetamat[i,:]
       propltheta = currentltheta + cholesky(lthetavar).L*randn(3*ncomponents)

       proptheta = exp.(reshape(propltheta, 3, ncomponents))

       nngp!(Bprop, Fprop, Border, nb, data.loc, data.time, proptheta)

       logpostprop = -0.5*( norm( (Bprop*w)./sqrt.(Fprop) )^2 + sum(log.(Fprop)) )
       logpost = -0.5*( norm( (B*w)./sqrt.(F) )^2 + sum(log.(F)) )

       logpriorprop = 0.0
       logprior = 0.0

       for i in 1:3
        for j in 1:ncomponents
            logpriorprop += gammaldens(proptheta[i,j], thetapriors.mu[i,j], thetapriors.var[i,j])
            logprior += gammaldens(theta[i,j], thetapriors.mu[i,j], thetapriors.var[i,j])
        end
       end

       acceptprob = exp.(logpostprop + logpriorprop + sum(propltheta) - logpost - logprior - sum(currentltheta))

       accepttheta = rand(1)[1] < acceptprob

       if accepttheta
            theta .= copy(proptheta)
            lthetamat[i+1,:] = copy(propltheta)
            B.nzval .= Bprop.nzval
            F .= Fprop
       else
            lthetamat[i+1,:] = copy(currentltheta)
       end



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