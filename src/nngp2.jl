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

function nngp2(neighbors::Matrix{Int64}, loc::AbstractArray, time::AbstractArray, theta::AbstractArray)

    local n = size(loc,1)
    local m = size(neighbors, 2)
    ncomponents = size(theta, 2)

    local Bnnz = sum(1:(m+1)) + (n - m - 1)*(m+1)
    local Bvals = zeros(Bnnz)
    Bvals[1] = 1

    local Brows = zeros(Int64, Bnnz)
    local Bcols = zeros(Int64, Bnnz)
    Brows[1] = 1 
    Bcols[1] = 1

    local Fvals = zeros(n)
    Fvals[1] = sum(theta[1,:].^2)

    curInd = 1

    @views for i in 2:(m+1)

        indi = 1:(i-1)

        mi = i - 1

        ds = pairwise(Euclidean(), loc[indi,:], dims = 1)
        dt = pairwise(Euclidean(), time[indi,:], dims = 1)
        dss = pairwise(Euclidean(), loc[indi,:], loc[[i],:], dims = 1)
        dtt = pairwise(Euclidean(), time[indi,:], time[[i],:], dims = 1)

        rho = expcord(ds, dt, theta)
        k = expcord(dss, dtt, theta)
        
        cholesky!(rho) 

        ldiv!(UpperTriangular(rho)', k)

        Fvals[i] = sum(theta[1,:].^2) - dot(k, k)

        ldiv!(UpperTriangular(rho), k)

        Bvals[(curInd+1):(curInd + mi + 1)] = [1; -k]
        Brows[(curInd+1):(curInd + mi + 1)] .= i
        Bcols[(curInd+1):(curInd + mi + 1)] = [i; indi]

        curInd += mi + 1

    end

    # Allocate arrays for computation within loop
    local rho = zeros(m, m)
    local ds = zeros(m, m)
    local dt = zeros(m,m)
    local k = zeros(m,1)
    local dss = zeros(m,1)
    local dtt = zeros(m,1)



    @views for i in (m+2):n
        
        indi = neighbors[i - m - 1,:]

        pairwise!(ds, Euclidean(), loc[indi,:], dims = 1)
        pairwise!(dt, Euclidean(), time[indi,:], dims = 1)
        pairwise!(dss, Euclidean(), loc[indi,:], loc[[i],:], dims = 1)
        pairwise!(dtt, Euclidean(), time[indi,:], time[[i],:], dims = 1)

        expcord!(rho, ds, dt, theta)
        expcord!(k, dss, dtt, theta)
        
        cholesky!(rho) 

        ldiv!(UpperTriangular(rho)', k)

        Fvals[i] = sum(theta[1,:].^2) - dot(k, k)

        ldiv!(UpperTriangular(rho), k)

        Bvals[(curInd+1):(curInd + m + 1)] = [1; -k]
        Brows[(curInd+1):(curInd + m + 1)] .= i
        Bcols[(curInd+1):(curInd + m + 1)] = [i; indi]


        curInd += m + 1

    end

    B = sparse(Brows, Bcols, Bvals)

    Border = invperm(sortperm( @.(Bcols + ( Brows ./ (n+1) ))))

    #println(cor(Bvals, B.nzval[Border]))

    return B, Fvals, Border

end