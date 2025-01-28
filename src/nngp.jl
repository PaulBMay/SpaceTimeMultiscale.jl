# Get nearest neighbor indices of a coordinate set. Based soley on spatial Euclidean, which is of dubious merit in some space-time scenarios.
function getneighbors(loc::Matrix{Float64}, m::Int64)

    local n = size(loc,1)

    local tree = KDTree(loc')

    skip_fun(i) = i >= ind

    local nb = zeros(Int64, n - m - 1, m)

    local ind = m+2

    for i in (m+2):n
        nb[i-m-1,:] = knn(tree, loc[i,:], m, false, skip_fun)[1]
        ind += 1
    end

    return nb

end


##########################
# Get NNGP mats.
# Spatial and Space-time versions, getting the B sparse matrix and F diagonal vector.
# There are in place versions, taking advantage of the consistent sparse structure across all covariance parameters.
# These in place versions require an ordering vector (returned by the not in place) to map the dense n x m covariance to the non-zero entries of the sparse matrix.
##########################

function nngp(neighbors::Matrix{Int64}, loc::AbstractArray, time::AbstractArray, theta::AbstractArray)

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

        rho = zeros(mi, mi)
        k = zeros(mi, 1)

        for component in 1:ncomponents # loop thru components, adding...

            rho +=  (theta[1, component]^2)*expCor(loc[indi,:], theta[2, component], time[indi,:], theta[3, component])

            k += (theta[1, component]^2)*expCor(loc[indi,:], loc[[i],:], theta[2, component], time[indi,:], time[[i],:], theta[3, component])

        end

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
    local rhodump = zeros(m, m)
    local T = zeros(m,m)
    local k = zeros(m,1)
    local kdump = zeros(m,1)
    local Tcross = zeros(m,1)



    @views for i in (m+2):n
        
        indi = neighbors[i - m - 1,:]

        fill!(rho, 0.0) # reset values to zero 
        fill!(k, 0.0)

        for component in 1:ncomponents # loop thru components, adding...

            expCor!(rhodump, T, loc[indi,:], theta[2, component], time[indi,:], theta[3, component])
            rhodump .*= theta[1, component]^2
            rho += rhodump 

            expCor!(kdump, Tcross, loc[indi,:], loc[[i],:], theta[2, component], time[indi,:], time[[i],:], theta[3, component])
            kdump .*= theta[1, component]^2
            k += kdump
            
        end

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

function nngp!(B::SparseMatrixCSC, Fvals::Vector{Float64}, Border::Vector{Int64}, neighbors::Matrix{Int64}, loc::AbstractArray, time::AbstractArray, theta::AbstractArray)

    n = size(loc,1)
    m = size(neighbors, 2)
    ncomponents = size(theta, 2)


    curInd = 1

    Fvals[1] = sum(theta[1,:].^2)

    @views for i in 2:(m+1)

        indi = 1:(i-1)

        mi = i - 1

        rho = zeros(mi, mi)
        k = zeros(mi, 1)

        for component in 1:ncomponents # loop thru components, adding...

            rho +=  (theta[1, component]^2)*expCor(loc[indi,:], theta[2, component], time[indi,:], theta[3, component])

            k += (theta[1, component]^2)*expCor(loc[indi,:], loc[[i],:], theta[2, component], time[indi,:], time[[i],:], theta[3, component])
            
        end

        cholesky!(rho) 

        ldiv!(UpperTriangular(rho)', k)

        Fvals[i] = sum(theta[1,:].^2) - dot(k, k)

        ldiv!(UpperTriangular(rho), k)

        valIndex = Border[(curInd+1):(curInd + mi + 1)]

        B.nzval[valIndex] .= [1; -k]

        curInd += mi + 1

    end

    # Allocate arrays for computation within loop
    local rho = zeros(m, m)
    local rhodump = zeros(m, m)
    local T = zeros(m,m)
    local k = zeros(m,1)
    local kdump = zeros(m,1)
    local Tcross = zeros(m,1)


    @views for i in (m+2):n
        
        indi = neighbors[i - m - 1,:]

        fill!(rho, 0.0) # reset values to zero 
        fill!(k, 0.0)

        for component in 1:ncomponents # loop thru components, adding...

            expCor!(rhodump, T, loc[indi,:], theta[2, component], time[indi,:], theta[3, component])
            rhodump .*= theta[1, component]^2
            rho += rhodump 

            expCor!(kdump, Tcross, loc[indi,:], loc[[i],:], theta[2, component], time[indi,:], time[[i],:], theta[3, component])
            kdump .*= theta[1, component]^2
            k += kdump
            
        end

        cholesky!(rho) 

        ldiv!(UpperTriangular(rho)', k)

        Fvals[i] = sum(theta[1,:].^2) - dot(k, k)

        ldiv!(UpperTriangular(rho), k)

        valIndex = Border[(curInd+1):(curInd + m + 1)]

        B.nzval[valIndex] .= [1; -k]

        curInd += m + 1

    end

    return nothing

end

#######
# Similar, but for prediction
########

function getpredneighbors(loc, locpred, m)

    npred = size(locpred, 1)

    tree = KDTree(loc')

    nbp = hcat(
        knn(tree, locpred', m, false)[1]...
    )'

    return Matrix(nbp)

end

function nngppred(neighbors::Matrix{Int64}, loc::AbstractArray, time::AbstractArray, locpred::AbstractArray, timepred::AbstractArray, theta::AbstractArray)

    local n = size(loc,1)
    local npred = size(locpred, 1)
    local m = size(neighbors, 2)

    local Bnnz = npred*m
    local Bvals = zeros(Bnnz)

    local Brows = repeat(1:npred, inner = m)
    local Bcols = vec(neighbors')

    local Fvals = zeros(npred)

    # Allocate arrays for computation within loop
    local rho = zeros(m, m)
    local rhodump = zeros(m, m)
    local T = zeros(m,m)
    local k = zeros(m,1)
    local kdump = zeros(m,1)
    local Tcross = zeros(m,1)

    @views for i in 1:npred
        
        indi = neighbors[i,:]

        fill!(rho, 0.0) # reset values to zero 
        fill!(k, 0.0)

        for component in 1:ncomponents # loop thru components, adding...

            expCor!(rhodump, T, loc[indi,:], theta[2, component], time[indi,:], theta[3, component])
            rhodump .*= theta[1, component]^2
            rho += rhodump 

            expCor!(kdump, Tcross, loc[indi,:], locpred[[i],:], theta[2, component], time[indi,:], timepred[[i],:], theta[3, component])
            kdump .*= theta[1, component]^2
            k += kdump
            
        end

        cholesky!(rho) 

        ldiv!(UpperTriangular(rho)', k)

        Fvals[i] = sum(theta[1,:].^2) - dot(k, k)

        ldiv!(UpperTriangular(rho), k)

        Bvals[((i-1)*m + 1):(i*m)] .= k

    end

    B = sparse(Brows, Bcols, Bvals, np, n)

    Border = invperm(sortperm( @.(Bcols + ( Brows ./ (np+1) ))))

    return B, Fvals, Border


end

function nngppred!(B::SparseMatrixCSC, Fvals::Vector{Float64}, Border::Vector{Int64}, neighbors::Matrix{Int64}, loc::AbstractArray, time::AbstractArray, locpred::AbstractArray, timepred::AbstractArray, theta::AbstractArray)

    local n = size(loc,1)
    local npred = size(locp, 1)
    local m = size(neighbors, 2)

    # Allocate arrays for computation within loop
    local rho = zeros(m, m)
    local rhodump = zeros(m, m)
    local T = zeros(m,m)
    local k = zeros(m,1)
    local kdump = zeros(m,1)
    local Tcross = zeros(m,1)

    @views for i in 1:npred
        
        indi = neighbors[i,:]

        fill!(rho, 0.0) # reset values to zero 
        fill!(k, 0.0)

        for component in 1:ncomponents # loop thru components, adding...

            expCor!(rhodump, T, loc[indi,:], theta[2, component], time[indi,:], theta[3, component])
            rhodump .*= theta[1, component]^2
            rho += rhodump 

            expCor!(kdump, Tcross, loc[indi,:], locpred[[i],:], theta[2, component], time[indi,:], timepred[[i],:], theta[3, component])
            kdump .*= theta[1, component]^2
            k += kdump
           
        end

        cholesky!(rho) 

        ldiv!(UpperTriangular(rho)', k)

        Fvals[i] = sum(theta[1,:].^2) - dot(k, k)

        ldiv!(UpperTriangular(rho), k)

        B.nzval[Border[((i-1)*m + 1):(i*m)]] .= k

    end

    return nothing

end