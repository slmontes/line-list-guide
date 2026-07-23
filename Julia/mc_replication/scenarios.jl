# Scenario runners for the Monte-Carlo replication.
#
# Each `*_run_once(seed) -> Vector{NamedTuple}` returns long-format rows
# (seed, method, metric, estimate, reference) for one replication. `SCENARIOS`
# is the registry the combined driver (run_all.jl) iterates.
#
# DDSA pipeline of each scenario only. Baseline geometry is :onset
# The `simulist` baseline is a single fixed R-generated and it was not MC replicated.
#
# Each runner emits the reference itself as a method (bias 0) — a built-in
# per-scenario sanity check that the paired reference logic is correct.

using DDSALineLists
using DataFrames
using Dates
using Distributions
using Random
using Statistics: mean

const _SCEN_HERE = @__DIR__
include(joinpath(_SCEN_HERE, "..", "shared", "fit_helpers.jl"))   # fit_lognormal_pcd, subsample_linelist

# Reduced per-replication sampling: we only need the posterior-median point per fit
# The replication spread supplies the uncertainty. 
const MC_NSAMPLES = 500
const MC_NCHAINS  = 1

_ddsa_params(; kwargs...) = DDSAParams(; β = 0.6, γ = 0.4, ρ = 0.005, N = 30_000, nsteps = 200, kwargs...)

# Fit with MC sampling settings (default D = max(delay)+2).
_mcfit(delays; pwindow = ones(length(delays)), seed::Int) =
    fit_lognormal_pcd(delays; pwindow = pwindow,
        n_samples = MC_NSAMPLES, n_chains = MC_NCHAINS, seed = seed)

# median/mean/sd rows for one fit against a matched reference fit.
function _metric_rows(seed, method, est, ref)
    [(seed = seed, method = method, metric = "median", estimate = est.median[1], reference = ref.median[1]),
     (seed = seed, method = method, metric = "mean",   estimate = est.mean[1],   reference = ref.mean[1]),
     (seed = seed, method = method, metric = "sd",     estimate = est.sd[1],     reference = ref.sd[1])]
end

# ==================================================================================
# recall bias
# ==================================================================================
const RECALL_P_LEVELS = [0.0, 0.1, 0.4]
const RECALL_PF_FIX    = 0.4    # forgetting level at which the correction is demonstrated
const RECALL_VAL_FRAC  = 0.30   # share of cases with a record-confirmed onset (assumed exact here)
const RECALL_N_IMP     = 15     # multiple-imputation draws for the validation correction

function _cap_at_admission!(ll::DataFrame)
    ll.date_onset_recalled = [
        ismissing(ll.date_onset_recalled[i]) || ismissing(ll.date_admission[i]) ?
            ll.date_onset_recalled[i] :
            min(ll.date_onset_recalled[i], ll.date_admission[i])
        for i in axes(ll, 1)]
    return ll
end

_recall_delays(ll) = Int[Dates.value(ll.date_admission[i] - ll.date_onset_recalled[i])
    for i in axes(ll, 1)
    if !ismissing(ll.date_onset_recalled[i]) && !ismissing(ll.date_admission[i])]

# Pool several posterior fits (one per imputation) into one estimate NamedTuple.
function _recall_pool_fits(fits)
    med = reduce(vcat, f.median_samples for f in fits)
    mn  = reduce(vcat, f.mean_samples   for f in fits)
    sdv = reduce(vcat, f.sd_samples     for f in fits)
    μs = log.(med)
    σs = sqrt.(max.(2 .* (log.(mn) .- μs), 0.0))
    d  = LogNormal(median(μs), median(σs))
    q(x) = (median(x), quantile(x, 0.025), quantile(x, 0.975))
    return (n = fits[1].n, dist = d, median = q(med), mean = q(mn), sd = q(sdv),
            median_samples = med, mean_samples = mn, sd_samples = sdv)
end

# Validation-informed correction: learn the recall-shift distribution on a
# record-confirmed subsample and multiply-impute it for the remaining cases.
function _recall_validation_fit(ll_clean::DataFrame, seed::Int)
    ll = copy(ll_clean)
    add_recall_bias!(ll; p_forget = RECALL_PF_FIX, seed = seed + 100 + 3)
    _cap_at_admission!(ll)

    n = nrow(ll)
    recalled_delay = Int[Dates.value(ll.date_admission[i]     - ll.date_onset_recalled[i]) for i in 1:n]
    true_delay     = Int[Dates.value(ll.date_admission[i]     - ll.date_onset[i])          for i in 1:n]
    recall_shift   = Int[Dates.value(ll.date_onset_recalled[i] - ll.date_onset[i])         for i in 1:n]

    rng_val = MersenneTwister(seed + 777)
    val_mask = falses(n)
    val_mask[sample(rng_val, 1:n, round(Int, RECALL_VAL_FRAC * n); replace = false)] .= true
    val_shifts = recall_shift[val_mask]

    rng_imp = MersenneTwister(seed + 888)
    imp_fits = NamedTuple[]
    for m in 1:RECALL_N_IMP
        d = Int[val_mask[i] ? true_delay[i] :
                recalled_delay[i] + sample(rng_imp, val_shifts) for i in 1:n]
        push!(imp_fits, _mcfit(Float64.(filter(≥(0), d)); seed = seed + 2000 + m))
    end
    return _recall_pool_fits(imp_fits)
end

function recall_run_once(seed::Int)
    ll_clean = simulate_linelist_ddsa(_ddsa_params();
        reporting_delay_dist = Distributions.Gamma(3, 1),
        admi_delay_dist = LogNormal(1.5, 0.5), seed = seed)
    ll_clean = subsample_linelist(ll_clean, 500; seed = seed)
    fits = Dict{Float64, NamedTuple}()
    for (k, pf) in enumerate(RECALL_P_LEVELS)
        ll = copy(ll_clean)
        add_recall_bias!(ll; p_forget = pf, seed = seed + 100 + k)
        _cap_at_admission!(ll)
        fits[pf] = _mcfit(_recall_delays(ll); seed = seed + k)
    end
    ref = fits[0.0]
    rows = NamedTuple[]
    for pf in RECALL_P_LEVELS
        append!(rows, _metric_rows(seed, "P_forget=$pf", fits[pf], ref))
    end
    est_validation = _recall_validation_fit(ll_clean, seed)
    append!(rows, _metric_rows(seed, "Validation-corrected", est_validation, ref))
    return rows
end

# ==================================================================================
# imputation (MCAR 50%)
# ==================================================================================
_imp_asymp(ll) = [ismissing(a) ? false : a for a in ll.asymptomatic]

function _imp_apply_missing!(ll::DataFrame; seed::Int)
    rng = MersenneTwister(seed)
    m = rand(rng, nrow(ll)) .< 0.5
    allowmissing!(ll, :date_onset)
    ll[m, :date_onset] .= missing
    return ll
end

function _imp_truth(ll::DataFrame)
    asymp = _imp_asymp(ll); d = Float64[]; pw = Float64[]
    for i in axes(ll, 1)
        asymp[i] && continue
        adm = ll.date_admission[i]; on = ll.date_onset[i]
        (ismissing(adm) || ismissing(on)) && continue
        v = Dates.value(adm - on); v >= 0 || continue
        push!(d, v); push!(pw, 1.0)
    end
    return d, pw
end

function _imp_adhoc(ll::DataFrame)
    asymp = _imp_asymp(ll); d = Float64[]; pw = Float64[]
    for i in axes(ll, 1)
        asymp[i] && continue
        adm = ll.date_admission[i]; ismissing(adm) && continue
        on = ll.date_onset[i]; rep = ll.date_reporting[i]
        ismissing(on) && (ismissing(rep) ? continue : (on = rep))
        v = Dates.value(adm - on); v >= 0 || continue
        push!(d, v); push!(pw, 1.0)
    end
    return d, pw
end

function _imp_meandelay(ll::DataFrame)
    asymp = _imp_asymp(ll); gaps = Float64[]
    for i in axes(ll, 1)
        asymp[i] && continue
        on = ll.date_onset[i]; rep = ll.date_reporting[i]
        (ismissing(on) || ismissing(rep)) && continue
        push!(gaps, Float64(Dates.value(rep - on)))
    end
    md = Day(round(Int, mean(gaps)))
    d = Float64[]; pw = Float64[]
    for i in axes(ll, 1)
        asymp[i] && continue
        adm = ll.date_admission[i]; ismissing(adm) && continue
        on = ll.date_onset[i]; rep = ll.date_reporting[i]
        ismissing(on) && (ismissing(rep) ? continue : (on = rep - md))
        v = Dates.value(adm - on); v >= 0 || continue
        push!(d, v); push!(pw, 1.0)
    end
    return d, pw
end

function _imp_missmodel(ll::DataFrame)
    asymp = _imp_asymp(ll)
    keep = [!asymp[i] for i in axes(ll, 1)]
    sub = ll[keep, :]
    obs = collect(skipmissing(sub.date_onset))
    min_obs = minimum(obs)
    d = Float64[]; pw = Float64[]
    for i in axes(sub, 1)
        adm = sub.date_admission[i]; ismissing(adm) && continue
        on = sub.date_onset[i]; rep = sub.date_reporting[i]
        if !ismissing(on)
            v = Dates.value(adm - on); v >= 0 || continue
            push!(d, v); push!(pw, 1.0)
        else
            earliest = max(min_obs, adm - Day(56))
            latest = ismissing(rep) ? adm : max(adm, rep)
            v = Dates.value(adm - earliest); v >= 0 || continue
            push!(d, v); push!(pw, Float64(Dates.value(latest - earliest) + 1))
        end
    end
    return d, pw
end

function imputation_run_once(seed::Int)
    ll = simulate_linelist_ddsa_asympt(_ddsa_params(; α = 0.4);
        reporting_delay_dist = Distributions.Gamma(3, 1),
        admi_delay_dist = LogNormal(1.5, 0.5), seed = seed)
    ll = subsample_linelist(ll, 850; seed = seed)
    td, tpw = _imp_truth(ll)
    ref = _mcfit(td; pwindow = tpw, seed = seed)
    _imp_apply_missing!(ll; seed = seed + 7)
    ad, apw = _imp_adhoc(ll); mdd, mdpw = _imp_meandelay(ll); mm, mmpw = _imp_missmodel(ll)
    rows = NamedTuple[]
    append!(rows, _metric_rows(seed, "Truth", ref, ref))
    append!(rows, _metric_rows(seed, "Adhoc-report",       _mcfit(ad;  pwindow = apw,  seed = seed + 1), ref))
    append!(rows, _metric_rows(seed, "Mean-delay",         _mcfit(mdd; pwindow = mdpw, seed = seed + 3), ref))
    append!(rows, _metric_rows(seed, "Interval-censored",  _mcfit(mm;  pwindow = mmpw, seed = seed + 2), ref))
    return rows
end

# ==================================================================================
# informative missingness
# ==================================================================================
function _im_positions(dates)
    d0 = minimum(skipmissing(dates))
    days = Float64[Dates.value(dates[i] - d0) for i in eachindex(dates)]
    span = maximum(days) - minimum(days)
    u = span > 0 ? days ./ span : fill(0.5, length(days))
    return u, mean(u)
end

function _im_trend!(ll::DataFrame; slope::Float64)
    u, ubar = _im_positions(ll.date_onset); on = ll.date_onset; adm = ll.date_admission
    for i in axes(ll, 1)
        base = Dates.value(adm[i] - on[i])
        adm[i] = on[i] + Day(max(base + round(Int, slope * (u[i] - ubar)), 0))
    end
    return ll
end

_im_mask(pmiss; seed) = (rng = MersenneTwister(seed); [rand(rng) < pmiss[i] for i in eachindex(pmiss)])

function _im_cc_delays(ll, mask)
    on = ll.date_onset; adm = ll.date_admission; d = Float64[]
    for i in axes(ll, 1)
        mask[i] && continue
        v = Dates.value(adm[i] - on[i]); v >= 0 || continue
        push!(d, v)
    end
    return d
end

function _im_ipw(ll, mask; binwidth = 7)
    rep = ll.date_reporting; on = ll.date_onset; adm = ll.date_admission
    t0 = minimum(skipmissing(rep)); binof(x) = Dates.value(x - t0) ÷ binwidth
    tot = Dict{Int, Int}(); obs = Dict{Int, Int}()
    for i in axes(ll, 1)
        ismissing(rep[i]) && continue
        b = binof(rep[i]); tot[b] = get(tot, b, 0) + 1
        mask[i] || (obs[b] = get(obs, b, 0) + 1)
    end
    d = Float64[]; w = Float64[]
    for i in axes(ll, 1)
        mask[i] && continue
        v = Dates.value(adm[i] - on[i]); v >= 0 || continue
        b = binof(rep[i]); pobs = get(obs, b, 0) / tot[b]
        push!(d, v); push!(w, pobs > 0 ? 1.0 / pobs : 1.0)
    end
    return d, w
end

function informative_missingness_run_once(seed::Int)
    ll = simulate_linelist_ddsa(_ddsa_params();
        reporting_delay_dist = Distributions.Gamma(3, 1),
        admi_delay_dist = LogNormal(1.5, 0.5), seed = seed)
    ll = subsample_linelist(ll, 1000; seed = seed)
    _im_trend!(ll; slope = 14.0)
    r, _ = _im_positions(ll.date_reporting)
    base = Float64[Dates.value(ll.date_admission[i] - ll.date_onset[i]) for i in axes(ll, 1)]
    n = nrow(ll)
    mask_mcar  = _im_mask(fill(0.5, n); seed = seed + 1)
    mask_delay = _im_mask(clamp.(0.05 .+ 0.07 .* base, 0.0, 0.90); seed = seed + 2)
    mask_time  = _im_mask(clamp.(0.05 .+ 0.90 .* r, 0.0, 0.97); seed = seed + 3)
    ref = _mcfit(_im_cc_delays(ll, falses(n)); seed = seed)
    dipw, wipw = _im_ipw(ll, mask_time)
    est_ipw = fit_lognormal_pcd(dipw; pwindow = ones(length(dipw)), weights = wipw,
        n_samples = MC_NSAMPLES, n_chains = MC_NCHAINS, seed = seed + 13)
    rows = NamedTuple[]
    append!(rows, _metric_rows(seed, "Truth", ref, ref))
    append!(rows, _metric_rows(seed, "MCAR",            _mcfit(_im_cc_delays(ll, mask_mcar);  seed = seed + 1), ref))
    append!(rows, _metric_rows(seed, "Delay-dep MNAR",  _mcfit(_im_cc_delays(ll, mask_delay); seed = seed + 2), ref))
    append!(rows, _metric_rows(seed, "Time-varying",    _mcfit(_im_cc_delays(ll, mask_time);  seed = seed + 3), ref))
    append!(rows, _metric_rows(seed, "Time-varying + IPW", est_ipw, ref))
    return rows
end

# ==================================================================================
# uncertain dates
# ==================================================================================
_ud_exact(ll) = Float64[Dates.value(ll.date_admission[i] - ll.date_onset[i]) for i in axes(ll, 1)
    if !ismissing(ll.date_onset[i]) && !ismissing(ll.date_admission[i]) &&
       Dates.value(ll.date_admission[i] - ll.date_onset[i]) >= 0]

function _ud_binned(ll; mode::Symbol)
    d = Float64[]; pw = Float64[]
    for i in axes(ll, 1)
        lo = ll.date_onset_lower[i]; hi = ll.date_onset_upper[i]; adm = ll.date_admission[i]
        (ismissing(lo) || ismissing(hi) || ismissing(adm)) && continue
        width = Dates.value(hi - lo)
        ref, w = if mode === :censor
            lo, Float64(width + 1)
        elseif mode === :midpoint
            lo + Day(width ÷ 2), 1.0
        else
            lo, 1.0
        end
        v = Dates.value(adm - ref); v >= 0 || continue
        push!(d, v); push!(pw, w)
    end
    return d, pw
end

function uncertain_dates_run_once(seed::Int)
    ll = simulate_linelist_ddsa(_ddsa_params();
        reporting_delay_dist = Distributions.Gamma(3, 1),
        admi_delay_dist = LogNormal(1.5, 0.5), seed = seed)
    ll = subsample_linelist(ll, 500; seed = seed)
    add_calendar_week_uncertainty!(ll)   # fixed weekly grid: onset uniform within its week
    ref = _mcfit(_ud_exact(ll); seed = seed)
    lod, lopw = _ud_binned(ll; mode = :lower)
    midd, midpw = _ud_binned(ll; mode = :midpoint)
    cend, cenpw = _ud_binned(ll; mode = :censor)
    rows = NamedTuple[]
    append!(rows, _metric_rows(seed, "Truth", ref, ref))
    append!(rows, _metric_rows(seed, "Lower-bound", _mcfit(lod;  pwindow = lopw,  seed = seed + 1), ref))
    append!(rows, _metric_rows(seed, "Midpoint",    _mcfit(midd; pwindow = midpw, seed = seed + 3), ref))
    append!(rows, _metric_rows(seed, "Censored",    _mcfit(cend; pwindow = cenpw, seed = seed + 2), ref))
    return rows
end

# ==================================================================================
# symptom definitions
# ==================================================================================
function _sd_between(ll, col)
    on = ll[!, col]; adm = ll.date_admission; d = Float64[]
    for i in axes(ll, 1)
        (ismissing(on[i]) || ismissing(adm[i])) && continue
        v = Dates.value(adm[i] - on[i]); v >= 0 || continue
        push!(d, v)
    end
    return d
end

function _sd_pooled(ll, mild_col, severe_col; p_severe, seed)
    rng = MersenneTwister(seed); mild = ll[!, mild_col]; sev = ll[!, severe_col]; adm = ll.date_admission
    d = Float64[]
    for i in axes(ll, 1)
        ismissing(adm[i]) && continue
        on = rand(rng) < p_severe ? sev[i] : mild[i]; ismissing(on) && continue
        v = Dates.value(adm[i] - on); v >= 0 || continue
        push!(d, v)
    end
    return d
end

function symptom_definition_run_once(seed::Int)
    ll = simulate_linelist_phase_mild_severe(_ddsa_params();
        p_progression = 1.0, admi_delay_dist = LogNormal(1.5, 0.5), seed = seed)
    ll = subsample_linelist(ll, 500; seed = seed)
    ref = _mcfit(_sd_between(ll, :date_onset_severe); seed = seed)         # consistent
    rows = NamedTuple[]
    append!(rows, _metric_rows(seed, "Severe onset (reference)", ref, ref))
    append!(rows, _metric_rows(seed, "Mild onset (shifted)", _mcfit(_sd_between(ll, :date_onset_mild); seed = seed + 1), ref))
    append!(rows, _metric_rows(seed, "Mixed onset (50/50)",
        _mcfit(_sd_pooled(ll, :date_onset_mild, :date_onset_severe; p_severe = 0.5, seed = seed + 10); seed = seed + 2), ref))
    return rows
end

# ==================================================================================
# date heaping
# ==================================================================================
function date_heaping_run_once(seed::Int)
    ll = simulate_linelist_ddsa(_ddsa_params();
        reporting_delay_dist = Distributions.Gamma(3, 1),
        admi_delay_dist = LogNormal(1.5, 0.5), seed = seed)
    ll = subsample_linelist(ll, 500; seed = seed)
    t0 = minimum(ll.date_onset); half = 7 ÷ 2
    onset_day = Int[Dates.value(ll.date_onset[i] - t0) for i in axes(ll, 1)]
    heaped_day = round.(Int, onset_day ./ 7) .* 7
    adm = ll.date_admission
    truth = Float64[]; naive = Float64[]; cens = Float64[]; cens_pw = Float64[]
    for i in axes(ll, 1)
        a = Dates.value(adm[i] - t0)
        dt = a - onset_day[i];  dt >= 0 && push!(truth, dt)
        dn = a - heaped_day[i]; dn >= 0 && push!(naive, dn)
        dc = a - (heaped_day[i] - half)
        if dc >= 0
            push!(cens, dc); push!(cens_pw, 7.0)
        end
    end
    ref = _mcfit(truth; seed = seed)
    rows = NamedTuple[]
    append!(rows, _metric_rows(seed, "Truth", ref, ref))
    append!(rows, _metric_rows(seed, "Naive-heaped",     _mcfit(naive; seed = seed + 1), ref))
    append!(rows, _metric_rows(seed, "Censored-window",  _mcfit(cens; pwindow = cens_pw, seed = seed + 2), ref))
    return rows
end

# ==================================================================================
# reporting-scheme change
# ==================================================================================
_rsc_delays(ll) = Int[Dates.value(ll.date_admission[i] - ll.date_onset[i]) for i in axes(ll, 1)
    if !ismissing(ll.date_onset[i]) && !ismissing(ll.date_admission[i])]

function _rsc_change_day(ll, t0, frac)
    days = sort(Int[Dates.value(o - t0) for o in ll.date_onset if !ismissing(o)])
    days[clamp(ceil(Int, frac * length(days)), 1, length(days))]
end

function _rsc_split(ll, t0, t_change; sev_intercept, sev_slope, seed)
    rng = MersenneTwister(seed)
    days_since = Int[Dates.value(ll.date_onset[i] - t0) for i in axes(ll, 1)]
    ota = Int[Dates.value(ll.date_admission[i] - ll.date_onset[i]) for i in axes(ll, 1)]
    pre_mask = days_since .<= t_change
    p_sev = @. 1.0 / (1.0 + exp(-(sev_intercept - sev_slope * ota)))
    severe = [rand(rng) < p_sev[i] for i in axes(ll, 1)]
    post_kept = (.!pre_mask) .& severe
    return (pre = ll[pre_mask, :], post_severe = ll[post_kept, :], pooled = ll[pre_mask .| post_kept, :])
end

function reporting_scheme_run_once(seed::Int)
    p = _ddsa_params()
    ll = simulate_linelist_ddsa(p;
        reporting_delay_dist = Distributions.Gamma(3, 1),
        admi_delay_dist = LogNormal(1.5, 0.5), seed = seed)
    ll = subsample_linelist(ll, 1500; seed = seed)
    tchange = _rsc_change_day(ll, p.t0, 0.3)
    sp = _rsc_split(ll, p.t0, tchange; sev_intercept = 2.2, sev_slope = 0.7, seed = seed + 50)
    ref = _mcfit(Float64.(_rsc_delays(ll)); seed = seed)                # true (all cases)
    rows = NamedTuple[]
    append!(rows, _metric_rows(seed, "Truth", ref, ref))
    append!(rows, _metric_rows(seed, "Pre-change",         _mcfit(Float64.(_rsc_delays(sp.pre));         seed = seed + 1), ref))
    append!(rows, _metric_rows(seed, "Post-change-severe", _mcfit(Float64.(_rsc_delays(sp.post_severe)); seed = seed + 2), ref))
    append!(rows, _metric_rows(seed, "Naive-pooled",       _mcfit(Float64.(_rsc_delays(sp.pooled));      seed = seed + 3), ref))
    return rows
end

# ==================================================================================
# registry
# ==================================================================================
const SCENARIOS = [
    (name = "imputation",              run = imputation_run_once),
    (name = "informative_missingness", run = informative_missingness_run_once),
    (name = "recall_bias",             run = recall_run_once),
    (name = "uncertain_dates",         run = uncertain_dates_run_once),
    (name = "symptom_definition",      run = symptom_definition_run_once),
    (name = "date_heaping",            run = date_heaping_run_once),
    (name = "reporting_scheme_change", run = reporting_scheme_run_once),
]
