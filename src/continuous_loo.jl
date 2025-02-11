function continuous_loo(outdir::String, data::InputData)

    effectspath = joinpath(outdir, "effects.csv")
    paramspath = joinpath(outdir, "params.csv")

    n = length(data.y)
    p = size(data.X, 2)

    Dsgn = sparse_hcat(data.X, speye(n))

    effects = CSV.read(effectspath, Tables.matrix)
    beta = view(effects, :, 1:p)
    w = view(effects, :, (p+1):(p+n))
    nsamps = size(effects, 1)

    params = CSV.read(paramspath, DataFrame)
    tSq = params.tSq

    lpd = zeros(n)
    nll = zeros(nsamps)

    for i = ProgressBar(1:n)
        for j in 1:nsamps
            μ = data.X[i,:]'*beta[j,:] + w[j, i]
            nll[j] = 0.5*( (data.y[i] - μ)^2 / tSq[j] + log(tSq[j]))
        end
        lpd[i] = log(nsamps) - logsumexp(nll)
    end

    return lpd


end

function continuous_loo(outdir::String, data::InputData, group::Vector{Integer})

    effectspath = joinpath(outdir, "effects.csv")
    paramspath = joinpath(outdir, "params.csv")

    n = length(data.y)
    p = size(data.X, 2)
    ngroups = maximum(group)

    Dsgn = sparse_hcat(data.X, speye(n))

    effects = CSV.read(effectspath, Tables.matrix)
    beta = view(effects, :, 1:p)
    w = view(effects, :, (p+1):(p+n))
    nsamps = size(effects, 1)

    params = CSV.read(paramspath, DataFrame)
    tSq = params.tSq

    lpd = zeros(ngroups)
    nll = zeros(nsamps)

    for i = ProgressBar(1:ngroups)
        groupind = group .== i
        for j in 1:nsamps
            μ = data.X[groupind,:]*beta[j,:] + w[j, groupind]
            nll[j] =  0.5*sum( (data.y[groupind] - μ).^2 ./ tSq[j] .+ log(tSq[j]))
        end
        lpd[i] = log(nsamps) - logsumexp(nll)
    end

    return lpd


end