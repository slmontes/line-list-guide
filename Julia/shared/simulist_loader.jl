using CSV
using DataFrames: DataFrame
using Dates: Date

"""
    load_simulist_baseline(; seed=42, cache=true)

Run `shared/simulist_baseline.R` via `Rscript` and return a `DataFrame` of the
clean simulist line list. Caches the CSV in this directory unless `cache=false`.

Returns `nothing` (with a warning) if `Rscript` fails. Callers should branch
on the return value to fall back to a DDSA-only comparison.
"""
function load_simulist_baseline(; seed::Int = 42, cache::Bool = true)
    here = @__DIR__
    cache_path = joinpath(here, "simulist_baseline_seed$(seed).csv")
    script = joinpath(here, "simulist_baseline.R")

    if !(cache && isfile(cache_path))
        cmd = `Rscript $script $cache_path $seed`
        try
            run(cmd)
        catch e
            @warn "simulist baseline failed; skipping that branch" exception = e
            return nothing
        end
    end
    isfile(cache_path) || return nothing

    df = CSV.read(cache_path, DataFrame; types = Dict(
        :asymptomatic => Bool,
        :date_onset => Date,
        :date_onset_mild => Date,
        :date_onset_severe => Date,
        :date_admission => Date,
        :date_first_contact => Date,
        :date_last_contact => Date,
        :date_reporting => Date,
    ), missingstring = "", validate = false)
    return df
end
