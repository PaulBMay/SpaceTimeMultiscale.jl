function bernoulli_loo(outdir::String, data::InputData)

    effectspath = joinpath(outdir, "effects.csv")

    n = length(data.y)
    p = size(data.X, 2)

    effects = CSV.read(effectspath, Tables.matrix)
    beta = view(effects, :, 1:p)
    w = view(effects, :, (p+1):(p+n))
    nsamps = size(effects, 1)

    lpd = zeros(n)
    nll = zeros(nsamps)

    for i = ProgressBar(1:n)
        for j in 1:nsamps
            μ = softmax(data.X[i,:]'*beta[j,:] + w[j, i])
            nll[j] = -(data.y[i]*log(μ) + (1 - data.y[i])*log(1 - μ))
        end
        lpd[i] = log(nsamps) - logsumexp(nll)
    end

    return lpd


end


function bernoulli_loo(outdir::String, data::InputData, group::Vector{Integer})

    effectspath = joinpath(outdir, "effects.csv")

    n = length(data.y)
    p = size(data.X, 2)
    ngroups = maximum(group)

    effects = CSV.read(effectspath, Tables.matrix)
    beta = view(effects, :, 1:p)
    w = view(effects, :, (p+1):(p+n))
    nsamps = size(effects, 1)

    lpd = zeros(n)
    nll = zeros(nsamps)

    for i = ProgressBar(1:ngroups)
        groupind = group .== i
        for j in 1:nsamps
            μ = softmax.(data.X[groupind,:]*beta[j,:] + w[j, groupind])
            nll[j] = -sum(data.y[groupind] .* log.(μ) + (1 .- data.y[groupind]) .* log.(1 .- μ))
        end
        lpd[i] = log(nsamps) - logsumexp(nll)
    end

    return lpd


end