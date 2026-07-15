# Uncertain date ranges (week-bin onset uncertainty)
Sandra Montes (@slmontes)
2026-07-06

## The issue

Onset dates are often known only approximately. When a patient reports
symptom onset approximately 1 or 2 weeks prior, they are providing a
valid temporal interval rather than an incorrect date. A rigorous line
list records this as an interval, represented in this scenario as a
seven-day window. This inherent uncertainty does not constitute a bias
in itself, as the true onset date genuinely falls within the specified
window. Rather, bias can be introduced later on through the
methodological choices an analyst makes when resolving that interval
into the single discrete value required by a naive delay model.

This scenario aims to isolate two distinct analytical consequences.
First, collapsing the interval to an incorrect single date introduces a
location bias. For instance, resolving every onset to the beginning of
its respective week systematically overestimates the resulting delays.
Second, collapsing the interval to any single date, even one that is
accurate on average, discards the underlying uncertainty. This approach
results in an inaccurately precise delay distribution, with credible
intervals that are narrower than what the raw data support. As a result,
the size of the resulting error is significantly influenced by the
analytical method selected. False precision arises when inherent
uncertainty is discarded rather than being accounted for in the
statistical fit.

By binning each onset onto a weekly grid anchored to the report date,
the scenario can represent the temporal uncertainty rather than
structural measurement error. This binning procedure is applied across
both pipelines to evaluate four distinct methodological approaches for
handling interval data.

## Methods

We evaluate four approaches to resolving the seven-day onset window:

- **Exact dates available:** the delay is computed from the true
  `date_onset` with `pwindow = 1`, providing the clean-data reference.
- **Ad-hoc resolution (lower bound):** each onset is binned into a
  seven-day window aligned to `date_reporting` and then resolved to the
  lower bound of that window, again with `pwindow = 1`. Because the bin
  always contains the true onset, the lower bound places onset as early
  as possible and therefore inflates the delay the most, making this the
  worst-case resolution.
- **Ad-hoc resolution (midpoint):** the same bins are used, but each
  onset is resolved to the bin midpoint, again with `pwindow = 1`. This
  is the common pragmatic choice: it is roughly centred on average, but
  because it still commits to a single date it inflates the dispersion,
  adding a spurious within-week spread on top of the true delay
  variation.
- **Model with censoring:** the same bins are used, but rather than
  resolving to a single date the censored fit takes the full bin width
  (`pwindow = 7`).

Presenting both the lower-bound and midpoint resolutions makes clear
that the magnitude of the bias is largely an analytical choice, with the
lower bound representing the worst case, whereas the censoring approach
provides the principled correction.

The lower-bound and midpoint resolutions both set `pwindow = 1`,
instructing the fit to treat the resolved date as exact. The censoring
approach instead sets `pwindow = 7`, which places a uniform prior over
where within the week the true onset fell and allows the likelihood to
integrate across that window rather than committing to a single point;
this is the mechanism by which the date uncertainty is propagated into
the credible intervals (Charniga 2024). The week-binning itself is
performed by the package helper `add_onset_uncertainty!`, a port of the
R `be_uncertain_date_week` that bins the true onset onto the
report-aligned weekly grid and writes `date_onset_lower` and
`date_onset_upper`; the two `week_bin_delays` resolutions are the
analyst’s choices layered on that recorded interval.

Inference throughout uses `fit_lognormal_pcd`, a lognormal delay fit by
Hamiltonian Monte Carlo (`Turing.jl`) under a primary-event–censored
likelihood from `CensoredDistributions.jl` (Abbott et al. 2025), the
Julia counterpart of R’s `primarycensored` (Abbott et al. 2026). As in
the other scenarios we run two independently built line lists: the DDSA
mechanistic model (Julia) and a `simulist` clean line list (R) degraded
by the identical week-binning rule. Agreement between them is a
cross-check that the bias is a property of the uncertainty handling
rather than of one generator.

## Setup

``` julia
using Pkg
Pkg.instantiate()

using DDSALineLists
using DataFrames
using Dates
using Distributions
using Random

include(joinpath(@__DIR__, "..", "shared", "fit_helpers.jl"))
include(joinpath(@__DIR__, "..", "shared", "scenario_plots.jl"))
include(joinpath(@__DIR__, "..", "shared", "simulist_loader.jl"))

const SEED = 1234
const N_SUB = 500   # realistic surveillance sample size (primary fit cohort)
const FIG_DIR = abspath(joinpath(@__DIR__, "..", "..", "figures"))
const OUT_PATH = joinpath(FIG_DIR, "issue_uncertain_dates.png")
```

## Helpers

``` julia
# Truth: pwindow = 1, delays from true onset.
function exact_delays(ll::AbstractDataFrame)
    delays = Float64[]
    for i in axes(ll, 1)
        on = ll.date_onset[i]
        adm = ll.date_admission[i]
        (ismissing(on) || ismissing(adm)) && continue
        d = Dates.value(adm - on)
        d >= 0 || continue
        push!(delays, d)
    end
    return delays
end

# Build (delay, pwindow) from the week-bin interval written by
# `add_onset_uncertainty!`, under three analyst choices:
#   :lower    — resolve onset to the bin's LOWER bound, pwindow = 1. The worst
#               case: because the bin always contains the true onset, the lower
#               bound pushes onset earliest and inflates the delay most.
#   :midpoint — resolve onset to the bin MIDPOINT, pwindow = 1. The common
#               pragmatic choice; roughly unbiased in the delay center but still
#               treats a single date as exact (under-disperses).
#   :censor   — keep the whole bin: delay from the lower bound, pwindow = bin
#               width, so the censored fit integrates over the uncertainty.
function week_bin_delays(ll::AbstractDataFrame; mode::Symbol)
    delays = Float64[]
    pwindows = Float64[]
    for i in axes(ll, 1)
        lo = ll.date_onset_lower[i]
        hi = ll.date_onset_upper[i]
        adm = ll.date_admission[i]
        (ismissing(lo) || ismissing(hi) || ismissing(adm)) && continue
        width = Dates.value(hi - lo)
        ref, pw = if mode === :censor
            lo, Float64(width + 1)
        elseif mode === :midpoint
            lo + Day(width ÷ 2), 1.0
        else  # :lower
            lo, 1.0
        end
        d = Dates.value(adm - ref)
        d >= 0 || continue
        push!(delays, d)
        push!(pwindows, pw)
    end
    return delays, pwindows
end

function fit_pcd(delays; pwindow = ones(length(delays)), seed::Int)
    fit_lognormal_pcd(delays;
        pwindow = pwindow,
        D = (length(delays) > 0 ? maximum(delays) : 0.0) + 2.0,
        n_samples = 1000, n_chains = 2, seed = seed)
end
```

## DDSA branch

``` julia
p = DDSAParams(β = 0.6, γ = 0.4, ρ = 0.005, N = 30_000, nsteps = 200)
ll_ddsa = simulate_linelist_ddsa(p;
    reporting_delay_dist = Distributions.Gamma(3, 1),
    admi_delay_dist = LogNormal(1.5, 0.5),
    seed = SEED,
)
ll_ddsa = subsample_linelist(ll_ddsa, N_SUB; seed = SEED)
add_onset_uncertainty!(ll_ddsa)  # writes date_onset_lower / date_onset_upper

ddsa_exact = exact_delays(ll_ddsa)
ddsa_lower, ddsa_lower_pw = week_bin_delays(ll_ddsa; mode = :lower)
ddsa_mid,   ddsa_mid_pw   = week_bin_delays(ll_ddsa; mode = :midpoint)
ddsa_model, ddsa_model_pw = week_bin_delays(ll_ddsa; mode = :censor)

est_ddsa_exact = fit_pcd(ddsa_exact;                            seed = SEED)
est_ddsa_lower = fit_pcd(ddsa_lower; pwindow = ddsa_lower_pw,   seed = SEED + 1)
est_ddsa_mid   = fit_pcd(ddsa_mid;   pwindow = ddsa_mid_pw,     seed = SEED + 3)
est_ddsa_model = fit_pcd(ddsa_model; pwindow = ddsa_model_pw,   seed = SEED + 2)

estimates = [est_ddsa_exact, est_ddsa_lower, est_ddsa_mid, est_ddsa_model]
labels = ["DDSA: exact dates available",
          "DDSA: ad-hoc resolution (week-bin lower bound)",
          "DDSA: ad-hoc resolution (week-bin midpoint)",
          "DDSA: model with censoring (pwindow = 7)"]
```

## simulist branch

The `simulist` baseline is cached as
`analyses/shared/simulist_baseline_seed<N>.csv`. If R or `simulist` is
unavailable the branch is skipped and a DDSA-only figure is produced.

``` julia
ll_sim = load_simulist_baseline(seed = SEED)
if !isnothing(ll_sim)
    have = .!ll_sim.asymptomatic .& .!ismissing.(ll_sim.date_admission) .&
           .!ismissing.(ll_sim.date_onset) .& .!ismissing.(ll_sim.date_reporting)
    sub = ll_sim[have, :]
    sub = subsample_linelist(sub, N_SUB; seed = SEED)
    add_onset_uncertainty!(sub)  # same observation process as the DDSA branch
    println("simulist symptomatic admitted with onset+reporting: $(nrow(sub)) cases")

    sim_exact = exact_delays(sub)
    sim_lower, sim_lower_pw = week_bin_delays(sub; mode = :lower)
    sim_mid,   sim_mid_pw   = week_bin_delays(sub; mode = :midpoint)
    sim_model, sim_model_pw = week_bin_delays(sub; mode = :censor)

    push!(estimates, fit_pcd(sim_exact;                         seed = SEED + 10))
    push!(estimates, fit_pcd(sim_lower; pwindow = sim_lower_pw, seed = SEED + 11))
    push!(estimates, fit_pcd(sim_mid;   pwindow = sim_mid_pw,   seed = SEED + 13))
    push!(estimates, fit_pcd(sim_model; pwindow = sim_model_pw, seed = SEED + 12))
    push!(labels, "simulist: exact dates available")
    push!(labels, "simulist: ad-hoc resolution (week-bin lower bound)")
    push!(labels, "simulist: ad-hoc resolution (week-bin midpoint)")
    push!(labels, "simulist: model with censoring (pwindow = 7)")
else
    @warn "Skipping simulist branch — DDSA-only figure"
end
```

## Figure

The reference is the exact-dates fit on undegraded DDSA data.

``` julia
fig = comparison_figure(
    estimates, labels;
    truth = (meanlog = est_ddsa_exact.dist.μ, sdlog = est_ddsa_exact.dist.σ),
    title = "Uncertain date ranges (week-bin onset uncertainty)",
)
save(OUT_PATH, fig)
fig
```

<img
src="issue_uncertain_dates_files/figure-commonmark/cell-6-output-1.png"
width="1100" height="720" />

## Results

Compared to a clean-data reference with a median of 4.44 days, the
handling choices span nearly the entire range of possible errors.
Resolving each week to its earliest day places every onset too early and
overestimates the delay, at about 7.80 days in DDSA and 7.74 in
`simulist`, because the lower bound sits on average around three days
before the true onset within its seven-day window (somewhat more in this
case, as onsets fall more densely later in the week). Using the midpoint
for resolution reduces location bias significantly (about 4.59 days in
DDSA and 4.53 in `simulist`), as the midpoint is closer to the average
true onset within the week.

Treating the whole week as interval-censored keeps the bias small (about
4.76 days in DDSA, 4.83 in `simulist`) and, unlike either point
resolution, propagates the date uncertainty into wider credible
intervals. The censored fit sits slightly above the reference on the
median. Its advantage is not a smaller point bias but the honest
uncertainty it carries. This small upward offset arises from the
modelling convention that the primary event is uniform within its
window, whereas within a reported week the true onset actually follows
the epidemic curve and is denser toward the peak; we adopt the uniform
convention to match the `primarycensored` workflow (Charniga 2024).

In sum, the analyst largely determines the magnitude of error, which can
range from severe under careless resolution to negligible under
censoring. False precision arises when uncertainty is disregarded rather
than properly accounted for.

## Estimates

    ┌ Info: DDSA: exact dates available
    │   n = 500
    │   median = (4.442017797100446, 4.246014028033202, 4.636091205372196)
    │   mean = (5.006874883465136, 4.781662723798852, 5.250721320881024)
    └   sd = (2.6014689918029585, 2.356052158960175, 2.8883595025413475)
    ┌ Info: DDSA: ad-hoc resolution (week-bin lower bound)
    │   n = 500
    │   median = (7.803354330225558, 7.5516767226129895, 8.06551020697539)
    │   mean = (8.326220906236788, 8.054333942275218, 8.617884877638367)
    └   sd = (3.093522674630867, 2.8631933497125206, 3.390487574171817)
    ┌ Info: DDSA: ad-hoc resolution (week-bin midpoint)
    │   n = 496
    │   median = (4.589167243691662, 4.341674573133975, 4.827701479914788)
    │   mean = (5.499953244358915, 5.1902643224884555, 5.837128800206143)
    └   sd = (3.6348439835717765, 3.2602311148985637, 4.150960546578224)
    ┌ Info: DDSA: model with censoring (pwindow = 7)
    │   n = 500
    │   median = (4.7556131628554335, 4.47219391782731, 5.029345190302102)
    │   mean = (5.238283582318932, 4.971354986219881, 5.527846511435977)
    └   sd = (2.4172758315793774, 2.136352136541535, 2.7617164380661703)
    ┌ Info: simulist: exact dates available
    │   n = 500
    │   median = (4.336928858408264, 4.159397790115631, 4.537212956532551)
    │   mean = (4.8972705021548055, 4.691845815904351, 5.137942693955153)
    └   sd = (2.563261960617906, 2.3250074353672665, 2.8457462987237143)
    ┌ Info: simulist: ad-hoc resolution (week-bin lower bound)
    │   n = 500
    │   median = (7.742242324636665, 7.485110282977389, 7.994631991619452)
    │   mean = (8.247316113387658, 7.9717966280398205, 8.52501816300447)
    └   sd = (3.023593293117942, 2.7850724117787156, 3.307517075727327)
    ┌ Info: simulist: ad-hoc resolution (week-bin midpoint)
    │   n = 496
    │   median = (4.529838519708534, 4.298818104194343, 4.788898307335586)
    │   mean = (5.408432793065018, 5.113307036146151, 5.751832714518799)
    └   sd = (3.5187584316225853, 3.1324516139625467, 3.9857936077147813)
    ┌ Info: simulist: model with censoring (pwindow = 7)
    │   n = 500
    │   median = (4.832525376369251, 4.574696098396727, 5.094596560007334)
    │   mean = (5.250184177058628, 4.9836825223669665, 5.509427749809275)
    └   sd = (2.2187472214459656, 1.9733554825721258, 2.519868208339426)

<div id="refs" class="references csl-bib-body hanging-indent"
entry-spacing="0">

<div id="ref-CensoredDistributions_jl" class="csl-entry">

Abbott, Sam, Damon Bayer, Sam Brand, Michael DeWitt, and Joseph
Lemaitre. 2025. “CensoredDistributions.jl.”
<https://doi.org/10.5281/zenodo.18474652>.

</div>

<div id="ref-primarycensored" class="csl-entry">

Abbott, Sam, Sam Brand, James Mba Azam, Carl Pearson, Sebastian Funk,
and Kelly Charniga. 2026. *Primarycensored: Primary Event Censored
Distributions*. <https://doi.org/10.5281/zenodo.13632839>.

</div>

<div id="ref-charniga2024delays" class="csl-entry">

Charniga, Sang Woo AND Akhmetzhanov, Kelly AND Park. 2024. “Best
Practices for Estimating and Reporting Epidemiological Delay
Distributions of Infectious Diseases.” *PLOS Computational Biology* 20
(10): 1–21. <https://doi.org/10.1371/journal.pcbi.1012520>.

</div>

</div>
