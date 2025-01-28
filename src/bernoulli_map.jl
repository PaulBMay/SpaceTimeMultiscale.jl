function bernoulli_theta_nlp(ltheta::Vector, data::InputData, nb::Matrix{Int64}, thetapriors::NamedTuple, betapriors::NamedTuple, Dsgn::SparseMatrixCSC, B::SparseMatrixCSC, F::Vector, Border::Vector, Qpostchol::SparseArrays.CHOLMOD.Factor, tol::Float64, maxiter::Int64)

    theta = exp.(reshape(ltheta, 3, :))

    n = length(data.y)
    p = size(data.X, 2)
    ncomponents = size(theta, 2)
    neffects = n + p

    nngp!(B, F, Border, nb, data.loc, data.time, theta)

    Q = blockdiag(spdiagm(betapriors.prec), (B'*spdiagm(1 ./ F)*B))

    ####################
    # Newton Raphson to find posterior mode of the effects
    ######################

    effects = zeros(neffects)
    update = copy(effects)
    probs = softmax.(Dsgn*effects)
    Omega = spdiagm(probs.*(1 .- probs))
    Qpost = Q + Dsgn'*(Omega*Dsgn)
    cholesky!(Qpostchol, Hermitian(Qpost))
    grad = Dsgn'*(data.y - probs) - Q*effects
    grad[1:p] += betapriors.mu .* betapriors.prec
    effects += (Qpostchol \ grad)

    error = 2.0
    count = 0

    while (error > tol) && (count <= maxiter)

        probs .= softmax.(Dsgn*effects)
        Omega .= spdiagm(probs.*(1 .- probs))
        Qpost .= Q + Dsgn'*(Omega*Dsgn)
        cholesky!(Qpostchol, Hermitian(Qpost))
        grad .= Dsgn'*(data.y - probs) - Q*effects
        grad[1:p] += betapriors.mu .* betapriors.prec
        update .= effects + (Qpostchol \ grad)
        error = norm(effects - update) / norm(update)
        effects .= copy(update)

        count += 1

    end

    ##################
    # Laplace approximation of the log posterior
    ####################

    # p(y | w, θ)

    probs .= softmax.(Dsgn*effects)
    pos = data.y .== 1
    lly = sum(log.(probs[pos])) + sum(log.(1 .- probs[.!pos]))

    # p(w | θ)

    effects[1:p] -= betapriors.mu

    llw = -0.5*(effects'*Q*effects + sum(log.(F)) - sum(log.(betapriors.prec)) + neffects*log(2*pi))

    # p(w | y, θ)

    llwc = -0.5*( -logdet(Qpostchol) + neffects*log(2*pi) )

    # p(θ)

    lprior = 0.0
    for i in 1:3
        for j in 1:ncomponents
            lprior += gammaldens(theta[i,j], thetapriors.mu[i,j], thetapriors.var[i,j])
        end
    end

    # p(θ | y)

    lpost = lly + llw + lprior - llwc

    return -lpost

    

end


function bernoulli_map(theta::AbstractArray, data::InputData, m::Integer, thetapriors::NamedTuple, betapriors::NamedTuple; nr_tol = 1e-4, nr_maxiter = 30, f_tol = 1e-3, g_tol = 1e-3, alpha = 1e-6, show_trace = true, store_trace = false)

    n = length(data.y)

    local nb = getneighbors(data.loc, m)

    local B, F, Border = nngp(nb, data.loc, data.time, theta)

    local Dsgn = sparse_hcat(data.X, speye(n))

    local Qpost = blockdiag(spdiagm(betapriors.prec), (B'*spdiagm(1 ./ F)*B)) + Dsgn'*Dsgn
    local Qpostchol = cholesky(Hermitian(Qpost))

    local ltheta = log.(vec(theta))
    
    thetamin = optimize(t -> bernoulli_theta_nlp(t, data, nb, thetapriors, betapriors, Dsgn, B, F, Border, Qpostchol, nr_tol, nr_maxiter), 
        ltheta, BFGS(alphaguess = Optim.LineSearches.InitialStatic(alpha=alpha)), 
        Optim.Options(f_tol = f_tol, g_tol = g_tol, store_trace = store_trace, show_trace = show_trace, extended_trace = (show_trace || store_trace))
    )

    thetamap = exp.(reshape(Optim.minimizer(thetamin), 3, :))


    return thetamap, thetamin
    
end