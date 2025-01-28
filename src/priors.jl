function gammaldens(x, gammean, gamvar)

    rate = gammean/gamvar
    shape = gammean*rate 

    ldens = (shape - 1)*log(x) - rate*x

    return ldens
end

function plotgamma(mu, v, bounds; nsteps = 1000)

    rate = mu/v
    shape = mu*rate


    x = range(bounds[1], bounds[2], nsteps)

    dens = @. exp((shape - 1)*log(x) - rate*x)

    display(plot(x, dens))

    return nothing

end

