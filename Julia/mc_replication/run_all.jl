# Run every wired scenario through the Monte-Carlo replication
# engine, emit one combined summary table (and CSV) tagged by scenario, and
# generate the LaTeX table that supplementary.tex inputs.
#
# Run:  julia --project=Julia Julia/mc_replication/run_all.jl [n_rep] [scenario ...]
#   n_rep      = number of replications (default 20). Use 50+ for production.
#   scenario   = one or more scenario names to run in isolation (for debugging).
#                Omit to run every scenario in the registry.
#
#   Examples:
#     julia ... run_all.jl 50                     # all scenarios, 50 reps
#     julia ... run_all.jl 20 recall_bias         # just recall bias, 20 reps
#     julia ... run_all.jl 20 recall_bias imputation
#
# To add a scenario: define its runner and append to SCENARIOS in scenarios.jl.

const HERE = @__DIR__
include(joinpath(HERE, "scenarios.jl"))   # SCENARIOS registry + runners
using CSV
using DataFrames
using Statistics: mean, quantile

# Monte-Carlo replication engine ------------------------------------------------
#
# A "scenario runner" is a function `run_once(seed) -> Vector{NamedTuple}`, where
# each returned row is one (method, metric) estimate for that replication:
#
#     (seed, method::String, metric::String, estimate::Float64, reference::Float64)
#
# `estimate` is the fitted value under the degradation/handling `method`;
# `reference` is the matched clean-data (or no-degradation) value from the SAME
# seed, so bias is computed paired within replication and baseline sampling
# variation cancels. The engine runs `run_once` over `seeds`, then summarises
# each (method, metric) across replications: mean estimate with a Monte-Carlo
# 95% interval, and mean bias / percent-bias with their Monte-Carlo intervals.
#
# This isolates systematic bias (its Monte-Carlo mean and spread across
# datasets) from the within-fit posterior uncertainty a single realisation shows.

"""
    mc_replicate(run_once; seeds, verbose=true) -> (raw::DataFrame, summary::DataFrame)

Run `run_once(seed)` for every seed and summarise. `raw` is the long-format
per-replication table; `summary` has one row per (method, metric).
"""
function mc_replicate(run_once; seeds::AbstractVector{<:Integer}, verbose::Bool = true)
    rows = NamedTuple[]
    for (i, s) in enumerate(seeds)
        verbose && print("\r  replication $i / $(length(seeds)) (seed $s)   ")
        append!(rows, run_once(s))
    end
    verbose && println()
    raw = DataFrame(rows)
    raw.bias = raw.estimate .- raw.reference
    raw.pct_bias = 100 .* raw.bias ./ raw.reference

    r3(x) = round(x; digits = 3)
    r1(x) = round(x; digits = 1)
    ql(x) = quantile(x, 0.025)
    qh(x) = quantile(x, 0.975)

    summary = combine(groupby(raw, [:method, :metric]),
        nrow => :n_reps,
        :estimate => (x -> r3(mean(x))) => :mean_est,
        :estimate => (x -> r3(ql(x)))   => :est_lo,
        :estimate => (x -> r3(qh(x)))   => :est_hi,
        :bias     => (x -> r3(mean(x))) => :mean_bias,
        :pct_bias => (x -> r1(mean(x))) => :mean_pct,
        :pct_bias => (x -> r1(ql(x)))   => :pct_lo,
        :pct_bias => (x -> r1(qh(x)))   => :pct_hi,
    )
    return raw, summary
end

n_rep = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 20   # replications per scenario
requested = ARGS[2:end]                                # scenario-name filter (empty = all)

known = Dict(sc.name => sc for sc in SCENARIOS)
if isempty(requested)
    scenarios = SCENARIOS
else
    bad = filter(n -> !haskey(known, n), requested)
    isempty(bad) || error("unknown scenario(s): $(join(bad, ", ")). " *
                          "Known: $(join(keys(known), ", "))")
    scenarios = [known[n] for n in requested]
end

seeds = collect(1000:(1000 + n_rep - 1))
println("Monte-Carlo replication over $(length(scenarios)) scenario(s), n_rep = $n_rep seeds")

summaries = DataFrame[]
for sc in scenarios
    println("\n>>> scenario: $(sc.name)")
    _, summ = mc_replicate(sc.run; seeds = seeds)
    insertcols!(summ, 1, :scenario => sc.name)
    push!(summaries, summ)
end

combined = reduce(vcat, summaries)
sort!(combined, [:scenario, :metric, :method])

println("\n=== Combined Monte-Carlo summary ===")
show(combined, allrows = true, allcols = true)
println()

outdir = joinpath(HERE, "results")
isdir(outdir) || mkpath(outdir)
csvname = "mc_combined_summary_$(n_rep).csv"   # rep count baked in so runs don't overwrite each other
csvpath = joinpath(outdir, csvname)
CSV.write(csvpath, combined)
println("\nWrote results/$(csvname)")

# Regenerate the LaTeX table and its companion forest plot only on a full run; a
# filtered subset would write partial artefacts over the ones the paper inputs.
if isempty(requested)
    println("Regenerating LaTeX table via format_mc_table.jl ...")
    include(joinpath(HERE, "format_mc_table.jl"))   # defines format_mc_table
    format_mc_table(csvpath)
    println("Regenerating forest plot via forest_plot.jl ...")
    include(joinpath(HERE, "forest_plot.jl"))       # defines make_forest_plot
    make_forest_plot(csvpath)
else
    println("Filtered run — skipping LaTeX table and forest-plot regeneration. " *
            "Run without a scenario filter to refresh mc_summary_table.tex and figures/mc_forest.png.")
end
