function expcord(ds::AbstractArray, dt::AbstractArray, theta::AbstractArray)

    local rho = zeros(size(dt)) 

    for t in eachcol(theta)
        rho .+= @. t[1]^2 * exp.(-ds / t[2] - dt / t[3])
    end

    return rho

end

function expcord!(rho::AbstractArray, ds::AbstractArray, dt::AbstractArray, theta::AbstractArray)

    fill!(rho, 0.0)
    for t in eachcol(theta)
        rho .+= @. t[1]^2 * exp.(-ds / t[2] - dt / t[3])
    end

    return nothing

end

function nngppred(neighbors::Matrix{Int64}, loc::AbstractArray, time::AbstractArray, locpred::AbstractArray, timepred::AbstractArray, theta::AbstractArray)

    local n = size(loc,1)
    local npred = size(locpred, 1)
    local m = size(neighbors, 2)
    ncomponents = size(theta, 2)

    local Bnnz = npred*m
    local Bvals = zeros(Bnnz)

    local Brows = repeat(1:npred, inner = m)
    local Bcols = vec(neighbors')

    local Fvals = zeros(npred)

    # Save distances for reuse

    Ds = zeros(m^2, npred)
    Dt = zeros(m^2, npred)

    ds = zeros(m, npred)
    dt = zeros(m, npred)


    # Allocate arrays for computation within loop
    local rho = zeros(m, m)
    local k = zeros(m,1)

    @views for i in 1:npred
        
        indi = neighbors[i,:]

        fill!(rho, 0.0) # reset values to zero 
        fill!(k, 0.0)

        pairwise!(reshape(Ds[:,i],  m, m), Euclidean(), loc[indi,:], dims = 1)
        pairwise!(reshape(Dt[:,i], m, m), Euclidean(), time[indi,:], dims = 1)

        pairwise!(ds[:,i:i], Euclidean(), loc[indi,:], locpred[[i],:], dims = 1)
        pairwise!(dt[:,i:i], Euclidean(), time[indi,:], timepred[[i],:], dims = 1)

        expcord!(rho, reshape(Ds[:,i],  m, m), reshape(Dt[:,i],  m, m), theta)
        expcord!(k, ds[:,i], dt[:,i], theta)

        cholesky!(Symmetric(rho)) 

        ldiv!(UpperTriangular(rho)', k)

        Fvals[i] = sum(theta[1,:].^2) - dot(k, k)

        ldiv!(UpperTriangular(rho), k)

        Bvals[((i-1)*m + 1):(i*m)] .= k

    end

    B = sparse(Brows, Bcols, Bvals, npred, n)

    Border = invperm(sortperm( @.(Bcols + ( Brows ./ (npred+1) ))))

    return B, Fvals, Border, NNDists(Ds, Dt, ds, dt, m)


end

function nngppred!(B::SparseMatrixCSC, Fvals::Vector{Float64}, Border::Vector{Int64}, NND::NNDists, theta::AbstractArray)

    local npred, n = size(B)
    local m = NND.m

    # Allocate arrays for computation within loop
    local rho = zeros(m, m)
    local k = zeros(m,1)

    @views for i in 1:npred
        

        fill!(rho, 0.0) # reset values to zero 
        fill!(k, 0.0)

        expcord!(rho, reshape(NND.Ds[:,i],  m, m), reshape(NND.Dt[:,i],  m, m), theta)
        expcord!(k, NND.ds[:,i], NND.dt[:,i], theta)

        cholesky!(Symmetric(rho))

        ldiv!(UpperTriangular(rho)', k)

        Fvals[i] = sum(theta[1,:].^2) - dot(k, k)

        ldiv!(UpperTriangular(rho), k)

        B.nzval[Border[((i-1)*m + 1):(i*m)]] .= k

    end

    return nothing

end