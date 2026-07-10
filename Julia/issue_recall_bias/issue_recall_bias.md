# Recall bias (telescoping toward the report date)
Sandra Montes (@slmontes)
2026-07-06

## The issue

Recall bias occurs when symptom onset dates are derived from patient
self-reporting rather than extracted from clinical records. Because
individuals recall recent events more accurately than distant ones, they
tend to anchor vague dates to more recent markers (“I think it started
last weekend”), so a remembered onset drifts toward the present. The
magnitude of this drift increases with the true time elapsed since
symptom onset. A patient interviewed one day after onset is unlikely to
misremember, whereas a patient interviewed two weeks later may telescope
the reported date forward by several days. This shortens the apparent
interval between onset and subsequent events, and biases delay estimates
downward. This effect can be more pronounced in interview-based
surveillance and retrospective outbreak investigations, where onset
dates are reconstructed days or weeks after the fact.

Here, the telescoping effect is modelled as a cumulative function of a
per-day forgetting probability, p_forget, to evaluate its impact on two
specific intervals: onset-to-admission (the primary estimand of this
study) and onset-to-report. Inference is conducted using
`fit_lognormal_pcd`, which performs a lognormal delay fit via
Hamiltonian Monte Carlo in `Turing.jl`. This approach relies on a
primary-event censored likelihood from `CensoredDistributions.jl`
(Abbott et al. 2025), the Julia equivalent of the R package
`primarycensored` (Charniga 2024; Abbott et al. 2026).

This example is shown for the DDSA pipeline only.

## Methods

A baseline DDSA line list is simulated, followed by the application of
the `add_recall_bias!` function using three distinct forgetting
probabilities (0, 0.1, and 0.4). Each day within the interval between
true onset and reporting is independently dropped with probability
`p_forget`. This mechanism shifts the recalled onset toward the report
date, generating the telescoping effect described previously. Because
each of the $g$ days in the true onset-to-report gap is omitted
independently, the expected magnitude of the recall shift is
$p_{\text{forget}} \cdot g$. This scaling means that longer delays
telescope more than shorter ones, and the compression is not uniform
across cases. This is the same reason the effect grows with the length
of the reporting delay.

We show both onset-to-admission and onset-to-report delays, since recall
bias can affect both. It can shorten the apparent onset-to-report delay
by exactly the recall shift, and shortens the apparent
onset-to-admission delay by the same amount (admission is anchored to
the true clinical event, not to the remembered onset).

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

const SEED = 1234
const N_SUB = 500      # realistic surveillance sample size (primary fit cohort)
const FIG_DIR = abspath(joinpath(@__DIR__, "..", "..", "figures"))
const OUT_PATH = joinpath(FIG_DIR, "issue_recall_bias.png")
const OUT_PATH_EPI = joinpath(FIG_DIR, "issue_recall_bias_epicurve.png")
const P_FORGET_LEVELS = [0.0, 0.1, 0.4]

function delays_using(ll::DataFrame, onset_col::Symbol, target_col::Symbol)
    on = ll[!, onset_col]
    tgt = ll[!, target_col]
    Int[Dates.value(tgt[i] - on[i]) for i in axes(ll, 1)
        if !ismissing(on[i]) && !ismissing(tgt[i])]
end
```

## Simulate a clean DDSA line list

``` julia
p = DDSAParams(β = 0.6, γ = 0.4, ρ = 0.005, N = 30_000, nsteps = 200)
ll_clean = simulate_linelist_ddsa(
    p;
    reporting_delay_dist = Distributions.Gamma(3, 1),
    admi_delay_dist = LogNormal(1.5, 0.5),
    seed = SEED,
)
ll_clean = subsample_linelist(ll_clean, N_SUB; seed = SEED)
println("DDSA clean line list: $(nrow(ll_clean)) cases")
```

    DDSA clean line list: 500 cases

## Apply recall bias at each forgetting level and fit

``` julia
est_admission = NamedTuple[]
est_report = NamedTuple[]
labels = String[]
recalled_onsets = Dict{Float64, Vector{Any}}()  # for the epidemic-curve panel
for (k, pf) in enumerate(P_FORGET_LEVELS)
    ll = copy(ll_clean)
    add_recall_bias!(ll; p_forget = pf, seed = SEED + 100 + k)
    recalled_onsets[pf] = collect(ll.date_onset_recalled)
    push!(est_admission,
          fit_lognormal_pcd(delays_using(ll, :date_onset_recalled, :date_admission);
                              n_samples = 1000, n_chains = 2, seed = SEED + k))
    push!(est_report,
          fit_lognormal_pcd(delays_using(ll, :date_onset_recalled, :date_reporting);
                              n_samples = 1000, n_chains = 2, seed = SEED + k))
    push!(labels, "p_forget = $pf")
end
```

Truth is the no-recall (`p_forget = 0`) fit, produced by the same
`fit_lognormal_pcd` estimator as every other scenario. Defining “truth”
as the no-recall estimate (rather than the analytic data-generating
distribution) isolates the recall effect. Under this framework, the
`p_forget = 0` scenario aligns with the dashed line, and any deviation
in the other scenarios represents the effect of recall bias.

``` julia
const TRUTH_ADMISSION = (meanlog = est_admission[1].dist.μ,
                         sdlog = est_admission[1].dist.σ)
const TRUTH_REPORT = (meanlog = est_report[1].dist.μ,
                      sdlog = est_report[1].dist.σ)
```

## Figure: delay distributions

``` julia
fig = comparison_figure_two_outcomes(
    est_admission, est_report, labels;
    truth_a = TRUTH_ADMISSION,
    truth_b = TRUTH_REPORT,
    title = "Recall bias: telescoping toward the report date",
    outcome_titles = ("Onset-to-admission delay (days)",
                      "Onset-to-report delay (days)"),
    density_xlabels = ("onset-to-admission delay (days)",
                       "onset-to-report delay (days)"),
)
save(OUT_PATH, fig)
fig
```

<img src="issue_recall_bias_files/figure-commonmark/cell-6-output-1.png"
width="1100" height="540" />

## Figure: epidemic-curve distortion

``` julia
function daily_counts(dates, t0::Date, maxday::Int)
    c = zeros(Int, maxday + 1)
    for d in dates
        ismissing(d) && continue
        day = Dates.value(d - t0)
        0 <= day <= maxday && (c[day + 1] += 1)
    end
    return c
end

t0 = minimum(ll_clean.date_onset)
true_days = Int[Dates.value(d - t0) for d in ll_clean.date_onset]
recalled_days = vcat((Int[Dates.value(d - t0) for d in recalled_onsets[pf] if !ismissing(d)]
                      for pf in P_FORGET_LEVELS)...)
maxday = maximum(vcat(true_days, recalled_days))

true_c = daily_counts(ll_clean.date_onset, t0, maxday)
xhi = findlast(>(0), true_c) + 12

fig_epi = Figure(size = (900, 360), figure_padding = (12, 18, 8, 8))
ax = Axis(fig_epi[1, 1];
    xlabel = "epidemic day (since first true onset)",
    ylabel = "cases by onset day",
    title = "Epidemic-curve distortion under recall bias",
    titlealign = :left)
lines!(ax, 0:maxday, true_c; color = :black, linewidth = 2.5, label = "true onset")
palette = Makie.wong_colors()
for (k, pf) in enumerate(P_FORGET_LEVELS)
    pf == 0.0 && continue
    c = daily_counts(recalled_onsets[pf], t0, maxday)
    lines!(ax, 0:maxday, c; color = palette[k], linewidth = 2,
           label = "recalled onset, p_forget = $pf")
end
xlims!(ax, 0, xhi)
axislegend(ax; position = :rt, framevisible = false, labelsize = 11)
save(OUT_PATH_EPI, fig_epi)
fig_epi
```

<img src="issue_recall_bias_files/figure-commonmark/cell-7-output-1.png"
width="900" height="360" />


    --- Onset-to-admission ---
    ┌ Info: p_forget = 0.0
    │   n = 500
    │   median = (4.452245169531041, 4.24919261287088, 4.656405456011626)
    │   mean = (5.0178800436933955, 4.796498652570649, 5.246954981539215)
    └   sd = (2.6153090876290532, 2.3777750704127456, 2.881952723175394)
    ┌ Info: p_forget = 0.1
    │   n = 499
    │   median = (4.143612976597187, 3.9484799986614325, 4.3549932167782455)
    │   mean = (4.810534495410321, 4.56605680364824, 5.071787121903395)
    └   sd = (2.8332702469578646, 2.5557480990943415, 3.158209959027372)
    ┌ Info: p_forget = 0.4
    │   n = 482
    │   median = (3.1255213231339742, 2.916976307260446, 3.343245048839495)
    │   mean = (4.130703351772457, 3.8246636973122805, 4.46277015593449)
    └   sd = (3.5676724844332766, 3.087750320250356, 4.152128026619419)

    --- Onset-to-report ---
    ┌ Info: p_forget = 0.0
    │   n = 500
    │   median = (2.5943579052296517, 2.467163160782835, 2.7269930520949215)
    │   mean = (3.0014242883242996, 2.857202188792413, 3.1671391895580245)
    └   sd = (1.7475161352602944, 1.565235222723354, 1.9672506259541285)
    ┌ Info: p_forget = 0.1
    │   n = 500
    │   median = (2.344496710639792, 2.2218929544141, 2.4708134448966383)
    │   mean = (2.7667538857138703, 2.616656457199326, 2.926586218365955)
    └   sd = (1.7329886496923281, 1.5420671306874882, 1.967485958825094)
    ┌ Info: p_forget = 0.4
    │   n = 500
    │   median = (1.4033327069301116, 1.3006658710681707, 1.5105251823008432)
    │   mean = (1.8847973853629734, 1.7188549661251897, 2.0904886201595607)
    └   sd = (1.6920790561023802, 1.3801982734874712, 2.1195598820430663)

## Results

Relative to the `p_forget = 0` reference, increasing the forgetting
probability shortens both delay distributions. The onset-to-admission
median decreases from approximately 4.45 days to 4.14 days at
`p_forget = 0.1` and to 3.13 days at `p_forget = 0.4`, representing a
downward bias of roughly 30% at the highest evaluated rate. Over the
same range, the onset-to-report median decreases from approximately 2.59
to 1.41 days.

At `p_forget = 0.4`, both delays decrease by a similar absolute
magnitude of approximately 1.2 to 1.3 days. Because admission and report
dates are fixed administrative dates, shifting the recalled onset toward
the report date subtracts an equivalent amount of time from both
intervals. In relative terms, the onset-to-report delay is more
compressed (roughly 45% compared to 30%) because its baseline duration
is shorter, so the same absolute shift is a larger fraction of it.
Because telescoping occurs only within the interval between actual onset
and reporting, the observed distortion remains modest under the short
reporting delay simulated here, which has a mean gap of approximately 3
days. This distortion would be expected to scale with longer reporting
gaps.

Additionally, serial interval estimates, which depend directly on onset
timing, would be subject to a similar bias. However, this effect is not
quantified here because the simulated line list records individual cases
rather than linked transmission pairs between an infector and an
infectee.

This telescoping phenomenon also distorts the reconstructed epidemic
curve. As recalled onsets shift closer to the report date, the overall
onset distribution shifts later and flattens. Specifically, the apparent
growth phase is dampened, the decline phase is broadened, and at
`p_forget = 0.4`, the mean onset day shifts later by approximately one
day. This structural shift is consequential because epidemic curves
derived from recalled onsets are used to estimate growth rates and
$R_t$. Therefore, estimation models might interpret the slower,
later-peaking curve as a less transmissible outbreak. Similar to the
effect on delay distributions, this distortion is constrained by the
short reporting delay simulated here but would amplify as the true
onset-to-report gap increases.

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
