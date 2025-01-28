# Spatial plot function. No 'using Plots' in here, so will crash if you haven't loaded Plots.jl in the referencing scope.
function quiltplot(loc, z)
    display(scatter(loc[:,1], loc[:,2], zcolor = z, markerstrokewidth = 0, alpha = 0.8))
    return nothing
end

# Create sparse id matrix
function speye(n::Integer)
    return sparse(1:n,1:n, 1.0)
end

# Paste location rows into a single string. Janky tool to identify unique locations and location matches...
function loc2str(loc::AbstractMatrix{Float64})
    return string.(loc[:,1]).*("_".*string.(loc[:,2]))
end

function softmax(x)
    return 1/(1 + exp(-x))
end

function logsumexp(x)
    xmax = maximum(x)
    return xmax + log(sum(exp.(x .- xmax)))
end

function datasplit(data::InputData)

    pos = data.y .> 0
    z = 1*pos

    dataPos = InputData(data.y[pos], data.X[pos,:], data.loc[pos,:], data.time[pos,:])
    dataZ = InputData(z, data.X, data.loc, data.time)

    return dataZ, dataPos


end


function cvsplit(data::InputData, testprop::Real)

    local locunq = unique(data.loc, dims = 1)
    nunq = size(locunq, 1)
    local map2unq = indexin(loc2str(data.loc), loc2str(locunq))
    ntestunq = Integer(floor(testprop*nunq))
    testindunq = sample(1:nunq, ntestunq)

    testind = map2unq .∈ [testindunq]

    testdata = InputData(data.y[testind], data.X[testind,:], data.loc[testind,:], data.time[testind,:])
    traindata = InputData(data.y[.!testind], data.X[.!testind,:], data.loc[.!testind,:], data.time[.!testind,:])

    return traindata, testdata


end


function getpropvar(path::String, nuse::Integer)

    pars = CSV.read(path, Tables.matrix)
    nsamps, npars = size(pars)
    
    use = (nsamps - nuse + 1):nsamps
    
    propvar = (2.4^2 / npars)*cov( log.(pars[use,:]) ) 

    return propvar

end

function getlastsamp(path::String)

    pars = CSV.read(path, Tables.matrix)
    nsamps, npars = size(pars)

    lastsamp = pars[nsamps,:]

    return lastsamp

end

function getCI(samps::AbstractArray, alpha::Number)

    q1 = (1 - alpha)/2
    q2 = 1 - q1

    n = size(samps, 2)

    CI = zeros(n, 2)

    for i = 1:n
        CI[i,:] = quantile(samps[:,i], [q1, q2])
    end

    return CI

end