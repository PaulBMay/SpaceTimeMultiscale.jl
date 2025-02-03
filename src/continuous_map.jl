# Compute the MAP estimates. Unconstrained BFGS using the log parameterization.
function continuous_map(theta::AbstractArray, tSq::Number, data::InputData, m::Integer, thetapriors::NamedTuple, tSqprior::NamedTuple; f_tol = 1e-3, g_tol = 1e-3, alpha = 1e-6, show_trace = true, store_trace = false)
    

    n = size(data.y, 1) # sample size
    p = size(data.X, 2) # num predictors

    neighbors = getneighbors(data.loc, m)

    B,F,Border = nngp(neighbors, data.loc, data.time, theta)

    Dsgn = sparse_hcat(data.X, speye(n))

    betaprec = fill(0.01, p)

    Qpost = blockdiag(
        spdiagm(betaprec),
        B'*spdiagm(1 ./ F)*B
    ) + (1/tSq)*Dsgn'*Dsgn

    Qpostchol = cholesky(Hermitian(Qpost))

    initpars = log.( vcat(vec(theta), tSq) )
    
    thetaMin = optimize(ty -> continuous_theta_nlp(ty, data, neighbors, thetapriors, tSqprior, betaprec, Dsgn, B, F, Border, Qpostchol), 
                            initpars, BFGS(alphaguess = Optim.LineSearches.InitialStatic(alpha=alpha)), 
                            Optim.Options(f_tol = f_tol, g_tol = g_tol, store_trace = store_trace, show_trace = show_trace, extended_trace = (show_trace || store_trace)))

    
    
    pars = Optim.minimizer(thetaMin)

    theta = reshape( exp.( pars[1:(end-1)] ), 3, :)
    tSq = exp.(pars[end])

    return theta, tSq, thetaMin


end


# Objective function for the MAP. Using the log parametrization.
function continuous_theta_nlp(pars::Vector, data::InputData, nb::Matrix, thetapriors::NamedTuple, tSqprior::NamedTuple, betaprec::Vector, Dsgn::SparseMatrixCSC, B::SparseMatrixCSC, F::Vector, Border::Vector, Qpostchol::SparseArrays.CHOLMOD.Factor)

    local theta = reshape( exp.( pars[1:(end-1)] ), 3, :)
    tSq = exp.(pars[end])

    n = length(data.y)
    ncomponents = size(theta, 2)

    nngp!(B, F, Border, nb, data.loc, data.time, theta)

    local Qpost = blockdiag(
        spdiagm(betaprec),
        B'*spdiagm(1 ./ F)*B
     ) + (1/tSq)*Dsgn'*Dsgn

    cholesky!(Qpostchol, Hermitian(Qpost))

    local ysolve = (data.y ./ tSq) - (Dsgn*(Qpostchol \ (Dsgn'*data.y) ) ./ tSq^2) # Woodbury

    local sse = dot(data.y, ysolve)

    local priorldet = sum(log.(betaprec)) - sum(log.(F)) 

    local ldet = n*log(tSq) + logdet(Qpostchol) - priorldet # Woodbury

    local nll = 0.5*(sse + ldet)

    lprior = 0.0
    for i in 1:3
        for j in 1:ncomponents
            lprior += gammaldens(theta[i,j], thetapriors.mu[i,j], thetapriors.var[i,j])
        end
    end

    lprior += gammaldens(tSq, tSqprior.mu, tSqprior.var)

    local nlp = nll - lprior

    return nlp


end


function continuous_hessian(thetamap::AbstractArray, tSqmap::Number, data::InputData, m::Integer, thetapriors::NamedTuple, tSqprior::NamedTuple; nr_tol = 1e-4, nr_maxiter = 30)

    n = size(data.y, 1) # sample size
    p = size(data.X, 2) # num predictors

    neighbors = getneighbors(data.loc, m)

    B,F,Border = nngp(neighbors, data.loc, data.time, thetamap)

    Dsgn = sparse_hcat(data.X, speye(n))

    betaprec = fill(0.01, p)

    Qpost = blockdiag(
        spdiagm(betaprec),
        B'*spdiagm(1 ./ F)*B
    ) + (1/tSqmap)*Dsgn'*Dsgn

    Qpostchol = cholesky(Hermitian(Qpost))

    lparmap = log.( vcat(vec(thetamap), tSqmap)) 

    local Hess = FiniteDiff.finite_difference_hessian(t -> continuous_theta_nlp(t, data, neighbors, thetapriors, tSqprior, betaprec, Dsgn, B, F, Border, Qpostchol), lparmap)

    return inv(Hess)

end
