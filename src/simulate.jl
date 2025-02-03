function simulate_field(theta::AbstractArray, tSq::Number, beta::Number, loc::AbstractArray, time::AbstractArray, m::Integer)

    n = size(loc, 1)

    local nb = getneighbors(loc, m)

    local B,F,Border = nngp(nb, loc, time, theta)

    local w = LowerTriangular(B) \ (sqrt.(F) .* randn(n))

    return beta .+ w .+ sqrt(tSq)*randn(n)

end