using DataFrames: DataFrame, AbstractDataFrame, nrow
using Dates: Dates, Date, Day
using Distributions: LogNormal, fit_mle, mean, std, median, quantile, Normal, Uniform, truncated
using Random: MersenneTwister, seed!
using StatsBase: sample
using Turing: Turing, @model, NUTS, MCMCThreads, MCMCSerial
import Turing
import MCMCChains
using CensoredDistributions: double_interval_censored, weight

"""
    subsample_linelist(ll, n; seed=1)

Uniform random subsample of `n` cases (rows) from line list `ll`, without
replacement (returns `ll` unchanged if `n ≥ nrow(ll)`). Brings the *observed*
line list down to a realistic surveillance scale before fitting.

This changes only how many cases are OBSERVED — not the population size `N` or
the epidemic dynamics — so it is consistent with the survival-dynamical-system
(SDS) large-population limit (Khudabukhsh et al. 2019, Interface Focus): the
mean-field infection-time survival functions depend on the population `N` (kept
large, e.g. 30 000), while the observed sample size only sets statistical
variance. The draw is uniform so the propagation-of-chaos independence of the
sampled individuals is preserved. NB: never shrink `N` to get fewer cases —
that would break the mean-field limit; subsample the observations instead.
"""
function subsample_linelist(ll::AbstractDataFrame, n::Integer; seed::Int = 1)
    nrow(ll) <= n && return ll
    rng = MersenneTwister(seed)
    idx = sort!(sample(rng, 1:nrow(ll), n; replace = false))
    return ll[idx, :]
end

"""
    fit_lognormal(delays_days::AbstractVector{<:Real}; n_boot=1000, alpha=0.05, seed=1)

Maximum-likelihood fit of a `LogNormal` to a vector of positive delays (in days),
with bootstrap confidence intervals for the median, mean, and standard deviation.

Used for quick inference and as a sanity check against the Bayesian variant.
Returns a `NamedTuple` with `n`, `dist`, and `median`/`mean`/`sd` as
`(point, lo, hi)` tuples.
"""
function fit_lognormal(delays_days::AbstractVector{<:Real}; n_boot::Int = 1000,
                       alpha::Float64 = 0.05, seed::Int = 1)
    x = collect(skipmissing(delays_days))
    x = Float64.(filter(>(0), x))
    n = length(x)
    n ≥ 5 || error("fit_lognormal: need at least 5 positive observations, got $n")

    d = fit_mle(LogNormal, x)
    rng = MersenneTwister(seed)

    boot_med = zeros(n_boot)
    boot_mean = zeros(n_boot)
    boot_sd = zeros(n_boot)
    for b in 1:n_boot
        idx = sample(rng, 1:n, n; replace = true)
        db = fit_mle(LogNormal, x[idx])
        boot_med[b] = median(db)
        boot_mean[b] = mean(db)
        boot_sd[b] = std(db)
    end
    qlo, qhi = alpha / 2, 1 - alpha / 2
    return (
        n = n,
        dist = d,
        median = (median(d), quantile(boot_med, qlo), quantile(boot_med, qhi)),
        mean = (mean(d), quantile(boot_mean, qlo), quantile(boot_mean, qhi)),
        sd = (std(d), quantile(boot_sd, qlo), quantile(boot_sd, qhi)),
    )
end

# --- Bayesian inference -------------------------------------------------

Turing.@model function _lognormal_model(x)
    μ ~ Normal(0.0, 5.0)
    σ ~ truncated(Normal(0.0, 2.0); lower = 0.0)
    @. x ~ LogNormal(μ, σ)
end

"""
    fit_lognormal_bayes(delays_days::AbstractVector{<:Real};
                        n_samples=1000, n_chains=2, alpha=0.05, seed=1)

Bayesian fit of a `LogNormal` to a vector of positive delays via NUTS with
weakly-informative priors:

    μ ~ Normal(0, 5)
    σ ~ HalfNormal(2)

Returns the same `(n, dist, median, mean, sd)` shape as
[`fit_lognormal`](@ref) — where `median`/`mean`/`sd` are
`(posterior_median, lower_credible, upper_credible)` triples — plus three
extra fields containing the full posterior samples for ridge plotting:

- `median_samples`: posterior samples of `exp(μ)`
- `mean_samples`:   posterior samples of `exp(μ + σ²/2)`
- `sd_samples`:     posterior samples of the implied LogNormal sd

`dist` uses the posterior median of `(μ, σ)` for plotting the data-distribution
density curve.
"""
function fit_lognormal_bayes(delays_days::AbstractVector{<:Real};
                              n_samples::Int = 1000, n_chains::Int = 2,
                              alpha::Float64 = 0.05, seed::Int = 1)
    x = collect(skipmissing(delays_days))
    x = Float64.(filter(>(0), x))
    n = length(x)
    n ≥ 5 || error("fit_lognormal_bayes: need at least 5 positive observations, got $n")

    seed!(seed)
    chain = if n_chains == 1
        Turing.sample(_lognormal_model(x), NUTS(), n_samples; progress = false)
    else
        Turing.sample(_lognormal_model(x), NUTS(), MCMCSerial(),
                      n_samples, n_chains; progress = false)
    end

    μ_samples = vec(Array(chain[:μ]))
    σ_samples = vec(Array(chain[:σ]))

    median_samples = exp.(μ_samples)
    mean_samples = exp.(μ_samples .+ σ_samples .^ 2 ./ 2)
    sd_samples = sqrt.((exp.(σ_samples .^ 2) .- 1) .* exp.(2 .* μ_samples .+ σ_samples .^ 2))

    qlo, qhi = alpha / 2, 1 - alpha / 2
    μ_point = median(μ_samples)
    σ_point = median(σ_samples)
    d = LogNormal(μ_point, σ_point)

    return (
        n = n,
        dist = d,
        median = (median(median_samples), quantile(median_samples, qlo), quantile(median_samples, qhi)),
        mean = (median(mean_samples), quantile(mean_samples, qlo), quantile(mean_samples, qhi)),
        sd = (median(sd_samples), quantile(sd_samples, qlo), quantile(sd_samples, qhi)),
        median_samples = median_samples,
        mean_samples = mean_samples,
        sd_samples = sd_samples,
    )
end

"""
Integer onset-to-admission delays in days from `ll`, given column names. Skips
rows where either side is missing.
"""
function delays_days(ll::AbstractDataFrame; from::Symbol = :date_onset,
                     to::Symbol = :date_admission)
    a = ll[!, from]
    b = ll[!, to]
    Int[Dates.value(b[i] - a[i]) for i in axes(ll, 1)
        if !ismissing(a[i]) && !ismissing(b[i])]
end

# --- Primary-event-censored Bayesian inference (R primarycensored equivalent) ---

# Per-observation primary-event censoring + daily interval binning + right
# truncation at D. Mirrors R's `primarycensored::pcd_cmdstan_model`
# likelihood:
#   - For known onsets, pwindow = 1 (the within-day primary-event window).
#   - For interval-censored or missing onsets, pwindow widens to span the
#     uncertainty interval (e.g. 7 for week-bins, longer for missing onsets).
#   - D is the relative observation time (max delay + 2 by default).
Turing.@model function _lognormal_pcd_model(observed_delay::AbstractVector,
                                            pwindow::AbstractVector,
                                            counts::AbstractVector,
                                            D::AbstractVector,
                                            μ_prior::Normal,
                                            σ_prior::Normal)
    μ ~ μ_prior
    σ ~ truncated(σ_prior; lower = 0.0)
    base = LogNormal(μ, σ)
    @inbounds for i in eachindex(observed_delay)
        d = double_interval_censored(
            base;
            primary_event = Uniform(0.0, max(pwindow[i], 1e-8)),
            upper = D[i],
            interval = 1.0,
        )
        observed_delay[i] ~ weight(d, counts[i])
    end
end

"""
    fit_lognormal_pcd(observed_delay; pwindow=ones(...), D=max+2,
                       μ_prior=Normal(1.0, 1.0), σ_prior=Normal(0.5, 0.5),
                       n_samples=1000, n_chains=2, alpha=0.05, seed=1)

Bayesian fit of a `LogNormal` with primary-event censoring, daily interval
binning, and right-truncation at `D` — the Julia equivalent of R's
`primarycensored::pcd_cmdstan_model` workflow.

Per-observation `pwindow` lets each row carry its own primary-event window:
`1` for known integer-day onsets, wider for interval-censored or missing
ones (e.g. 7 for `be_uncertain_date_week` bins, or
`latest_onset - earliest_onset + 1` for unknowns).

`D` is the relative observation time used for right-truncation. It accepts
either a scalar (one truncation bound for the whole sample) or a vector
aligned with `observed_delay` (one bound per case). The per-case form is what
implements the *truncation correction* for real-time / snapshot analysis: at a
snapshot taken `T` days after the epidemic origin, a case with onset on day
`oᵢ` has been observable for only `Dᵢ = T − oᵢ` days, so its delay is
right-truncated at `Dᵢ`. Passing that vector lets the censored likelihood undo
the right-truncation; the scalar default (`max(observed_delay) + 2`) applies a
near-vacuous bound, i.e. no truncation correction.

`weights` optionally attaches a per-observation likelihood weight (default 1.0),
folded into the same weighted likelihood as the count-grouping so the weights
may be fractional — used for inverse-probability-weighted (survey-style)
corrections, e.g. up-weighting under-observed cases under time-varying
missingness.

Default priors: `meanlog ~ Normal(1, 1)`,`sdlog ~ Normal(0.5, 0.5)` (truncated positive).
The return value matches [`fit_lognormal_bayes`](@ref) so plotting code is interchangeable.
"""
function fit_lognormal_pcd(observed_delay::AbstractVector;
                            pwindow::AbstractVector = ones(length(observed_delay)),
                            D::Union{Real, AbstractVector, Nothing} = nothing,
                            weights::Union{AbstractVector, Nothing} = nothing,
                            μ_prior::Normal = Normal(1.0, 1.0),
                            σ_prior::Normal = Normal(0.5, 0.5),
                            n_samples::Int = 1000, n_chains::Int = 2,
                            alpha::Float64 = 0.05, seed::Int = 1)
    length(observed_delay) == length(pwindow) ||
        error("fit_lognormal_pcd: pwindow must align with observed_delay")
    # Optional per-observation likelihood weights (e.g. inverse-probability
    # weights for a survey-style correction). Default 1.0 each — i.e. plain
    # multiplicity. Weights are folded into the same weighted likelihood used
    # for the count-grouping, so they may be fractional.
    wfull = if weights === nothing
        ones(length(observed_delay))
    else
        length(weights) == length(observed_delay) ||
            error("fit_lognormal_pcd: weights must align with observed_delay")
        Float64.(collect(weights))
    end
    # Resolve D into a per-observation vector aligned with observed_delay.
    # `nothing` → the near-vacuous scalar default (no truncation correction);
    # a scalar → broadcast to every row; a vector → per-case truncation bound.
    Dfull = if D === nothing
        fill(Float64(maximum(skipmissing(observed_delay)) + 2), length(observed_delay))
    elseif D isa AbstractVector
        length(D) == length(observed_delay) ||
            error("fit_lognormal_pcd: D vector must align with observed_delay")
        Float64.(collect(D))
    else
        fill(Float64(D), length(observed_delay))
    end
    keep = [!ismissing(observed_delay[i]) && !ismissing(pwindow[i]) &&
            !ismissing(Dfull[i]) && observed_delay[i] >= 0 for i in eachindex(observed_delay)]
    x = Float64.(collect(observed_delay)[keep])
    pw = Float64.(collect(pwindow)[keep])
    Dv = Dfull[keep]
    wv = wfull[keep]
    n = length(x)
    n ≥ 5 || error("fit_lognormal_pcd: need at least 5 observations, got $n")

    # Group identical (delay, pwindow, D) triples and use a weighted likelihood. 
    # With a per-case D the triples are less degenerate (one group per distinct
    # onset day), but grouping still collapses cases sharing an onset day. The
    # group weight is the sum of the per-observation weights (counts when all
    # weights are 1), so fractional IPW weights pass through unchanged.
    counts = Dict{Tuple{Float64, Float64, Float64}, Float64}()
    for (xi, pwi, Di, wi) in zip(x, pw, Dv, wv)
        counts[(xi, pwi, Di)] = get(counts, (xi, pwi, Di), 0.0) + wi
    end
    keys_sorted = sort!(collect(keys(counts)))
    x_g = [k[1] for k in keys_sorted]
    pw_g = [k[2] for k in keys_sorted]
    D_g = [k[3] for k in keys_sorted]
    w_g = Float64[counts[k] for k in keys_sorted]

    seed!(seed)
    chain = if n_chains == 1
        Turing.sample(_lognormal_pcd_model(x_g, pw_g, w_g, D_g, μ_prior, σ_prior),
                      NUTS(), n_samples; progress = false)
    else
        Turing.sample(_lognormal_pcd_model(x_g, pw_g, w_g, D_g, μ_prior, σ_prior),
                      NUTS(), MCMCSerial(), n_samples, n_chains; progress = false)
    end

    μ_samples = vec(Array(chain[:μ]))
    σ_samples = vec(Array(chain[:σ]))

    median_samples = exp.(μ_samples)
    mean_samples = exp.(μ_samples .+ σ_samples .^ 2 ./ 2)
    sd_samples = sqrt.((exp.(σ_samples .^ 2) .- 1) .* exp.(2 .* μ_samples .+ σ_samples .^ 2))

    qlo, qhi = alpha / 2, 1 - alpha / 2
    μ_point = median(μ_samples)
    σ_point = median(σ_samples)
    d = LogNormal(μ_point, σ_point)

    return (
        n = n,
        dist = d,
        median = (median(median_samples), quantile(median_samples, qlo), quantile(median_samples, qhi)),
        mean = (median(mean_samples), quantile(mean_samples, qlo), quantile(mean_samples, qhi)),
        sd = (median(sd_samples), quantile(sd_samples, qlo), quantile(sd_samples, qhi)),
        median_samples = median_samples,
        mean_samples = mean_samples,
        sd_samples = sd_samples,
    )
end
