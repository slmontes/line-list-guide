# Forest plot of median percent bias with 2.5--97.5% Monte-Carlo intervals across
# scenarios — a visual companion to Table~\ref{tab:mc-replication}. PROVISIONAL:
# kept pending the review decision on whether the paper carries this figure.
#
# Run:  julia --project=Julia Julia/mc_replication/forest_plot.jl [in.csv] [out.png]
#   defaults: newest results/mc_combined_summary_<n_rep>.csv -> ../../figures/mc_forest.png
#
# Reuses latest_summary_csv / SCEN_ORDER / SCEN_LABEL from format_mc_table.jl.

const FP_HERE = @__DIR__
include(joinpath(FP_HERE, "format_mc_table.jl"))   # defines the helpers above (guarded main)
using CairoMakie

incsv  = length(ARGS) >= 1 ? ARGS[1] : latest_summary_csv()
outpng = length(ARGS) >= 2 ? ARGS[2] : joinpath(FP_HERE, "..", "..", "figures", "mc_forest.png")

df  = CSV.read(incsv, DataFrame)
med = df[df.metric .== "median", :]

# Plain-text scenario labels (strip the LaTeX escaping used in the table).
plain(s) = replace(String(s), "\\%" => "%", "\\_" => "_", "\\&" => "&")

# Build ordered rows: scenarios in presentation order, methods by |median %bias|
# ascending so the reference (0) leads each group, matching the table.
function build_rows(med)
    methods = String[]; xs = Float64[]; los = Float64[]; his = Float64[]; isref = Bool[]
    groups = Tuple{String,Int,Int}[]       # (scenario, first_row, last_row)
    row = 0
    for sc in SCEN_ORDER
        sub = med[med.scenario .== sc, :]
        isempty(sub) && continue
        first_row = row + 1
        refrows  = findall(r -> sub.mean_pct[r] == 0 && sub.pct_lo[r] == 0 && sub.pct_hi[r] == 0, 1:nrow(sub))
        restrows = sort(setdiff(1:nrow(sub), refrows); by = r -> abs(sub.mean_pct[r]))
        for i in vcat(refrows, restrows)
            row += 1
            push!(methods, String(sub.method[i]))
            push!(xs, sub.mean_pct[i]); push!(los, sub.pct_lo[i]); push!(his, sub.pct_hi[i])
            push!(isref, sub.mean_pct[i] == 0 && sub.pct_lo[i] == 0 && sub.pct_hi[i] == 0)
        end
        push!(groups, (sc, first_row, row))
    end
    return methods, xs, los, his, isref, groups
end

methods, xs, los, his, isref, groups = build_rows(med)
n = length(methods)

# Row n at the top so the first scenario appears first.
ys = Float64.(n:-1:1)
row_y(r) = n - r + 1                        # data-space y of ordered row r

xmin = minimum(los); xmax = maximum(his); span = max(xmax - xmin, 1.0)
labelx = xmin - 0.42 * span                 # x anchor for scenario captions

fig = Figure(size = (960, 26n + 130))
ax  = Axis(fig[1, 1];
    xlabel = "Median percent bias (%)",
    yticks = (ys, methods),
    ygridvisible = false,
    title = "Replicated median bias across scenarios (100 line lists, DDSA)")

vlines!(ax, [0]; color = (:black, 0.45), linestyle = :dash, linewidth = 1)

# Group separators and scenario captions.
for (k, (sc, a, b)) in enumerate(groups)
    k > 1 && hlines!(ax, [row_y(a) + 0.5]; color = (:gray, 0.25), linewidth = 0.8)
    text!(ax, labelx, (row_y(a) + row_y(b)) / 2; text = plain(SCEN_LABEL[sc]),
          align = (:left, :center), fontsize = 12, font = :bold, color = :gray25)
end

# Intervals then points; reference rows in grey, biased rows in colour.
errorbars!(ax, xs, ys, xs .- los, his .- xs; direction = :x,
           whiskerwidth = 6, color = :gray40, linewidth = 1.2)
scatter!(ax, xs, ys; color = [r ? :gray55 : :firebrick for r in isref], markersize = 9)

xlims!(ax, labelx - 0.02 * span, xmax + 0.06 * span)
ylims!(ax, 0.4, n + 0.6)

mkpath(dirname(outpng))
save(outpng, fig)
println("Wrote $outpng  ($n rows from $(basename(incsv)))")
