using CairoMakie
using Distributions: LogNormal, pdf, mean, std, quantile
using KernelDensity: kde

# --- internal helpers ----------------------------------------------------

# Draw a horizontal posterior ridge at row y_row: filled KDE of `samples`
# extending upward by at most `max_height` from the baseline.
function _draw_posterior_ridge!(ax, samples::AbstractVector, y_row::Real,
                                color; max_height::Real = 0.6,
                                fill_alpha::Real = 0.45)
    k = kde(samples)
    h = maximum(k.density)
    ys_norm = h > 0 ? (k.density ./ h) .* max_height : collect(k.density)
    band!(ax, k.x, fill(y_row, length(k.x)), y_row .+ ys_norm;
          color = (color, fill_alpha))
    lines!(ax, k.x, y_row .+ ys_norm; color = color, linewidth = 1.2)
    return nothing
end

# Compute a shared x-range for the median + mean panels using the 99%
# posterior intervals so the visible ridge tails sit comfortably inside.
function _shared_x_range(estimates, summaries; truth_vals = nothing)
    xs_all = Float64[]
    for s in summaries, e in estimates
        samples_field = Symbol(string(s) * "_samples")
        if hasproperty(e, samples_field)
            samples = getproperty(e, samples_field)
            push!(xs_all, quantile(samples, 0.005), quantile(samples, 0.995))
        else
            point, lo, hi = getproperty(e, s)
            push!(xs_all, lo, hi)
        end
    end
    if !isnothing(truth_vals)
        for s in summaries
            push!(xs_all, getproperty(truth_vals, s))
        end
    end
    xlo, xhi = extrema(xs_all)
    pad = 0.06 * (xhi - xlo)
    return xlo - pad, xhi + pad
end

# Render one outcome's three panels (density + 2 forests) into the given row.
function _render_outcome_row!(fig, row::Int, estimates::AbstractVector,
                              colors::AbstractVector, truth_dist;
                              density_xlabel::AbstractString)
    n = length(estimates)
    truth_vals = (
        median = quantile(truth_dist, 0.5),
        mean = mean(truth_dist),
    )

    ax_d = Axis(fig[row, 1];
        xlabel = density_xlabel,
        ylabel = "density",
        title = "Fitted LogNormal densities",
        titlealign = :left,
    )
    xlims!(ax_d, 0, 20)
    xs = range(0.01, 20; length = 600)
    lines!(ax_d, xs, pdf.(truth_dist, xs);
           color = :black, linestyle = :dash, linewidth = 3.5)
    for i in 1:n
        lines!(ax_d, xs, pdf.(estimates[i].dist, xs);
               color = colors[i], linewidth = 2)
    end

    summaries = [:median, :mean]
    sum_titles = ["median", "mean"]
    ys = collect(n:-1:1)
    xlo, xhi = _shared_x_range(estimates, summaries; truth_vals)

    for (j, s) in enumerate(summaries)
        ax = Axis(fig[row, j + 1];
            xlabel = "estimate (days)",
            title = sum_titles[j],
            titlealign = :left,
        )
        vlines!(ax, [getproperty(truth_vals, s)];
                color = :black, linestyle = :dash, linewidth = 3.5)
        for i in 1:n
            samples_field = Symbol(string(s) * "_samples")
            if hasproperty(estimates[i], samples_field)
                _draw_posterior_ridge!(ax,
                    getproperty(estimates[i], samples_field),
                    ys[i], colors[i])
                point = getproperty(estimates[i], s)[1]
                scatter!(ax, [point], [ys[i]]; color = colors[i], markersize = 7,
                         strokecolor = :white, strokewidth = 0.8)
            else
                point, lo, hi = getproperty(estimates[i], s)
                errorbars!(ax, [point], [ys[i]], [point - lo], [hi - point];
                           direction = :x, color = colors[i],
                           whiskerwidth = 8, linewidth = 2)
                scatter!(ax, [point], [ys[i]]; color = colors[i], markersize = 11,
                         strokecolor = :white, strokewidth = 1)
            end
        end
        xlims!(ax, xlo, xhi)
        ylims!(ax, 0.4, n + 0.9)
        hideydecorations!(ax; grid = false)
    end
    return nothing
end

# --- public API ----------------------------------------------------------

"""
    comparison_figure(estimates::AbstractVector, labels::AbstractVector;
                      truth::NamedTuple, title="")

Render a three-panel comparison figure:

- Left:   density curves of fitted LogNormal distributions, one per scenario,
          with truth as a black dashed line. Distributional spread is read
          off the curve widths.
- Middle: forest of *median* estimates with posterior densities (or 95%
          bootstrap CIs) drawn as horizontal ridges.
- Right:  forest of *mean* estimates, same.

Each estimate may carry posterior samples (`median_samples`, `mean_samples`)
from `fit_lognormal_bayes`; if present they are drawn as KDE ridges. If only
the `(point, lo, hi)` triples are present (`fit_lognormal` MLE), the row falls
back to a dot + errorbar.

Method labels are not drawn on the forest panels; instead a single horizontal
legend at the bottom of the figure maps each colour to its scenario.
"""
function comparison_figure(estimates::AbstractVector, labels::AbstractVector;
                           truth::NamedTuple, title::AbstractString = "")
    length(estimates) == length(labels) || error("estimates/labels length mismatch")
    n = length(estimates)
    palette = Makie.wong_colors()
    colors = [palette[mod1(i, length(palette))] for i in 1:n]
    truth_dist = LogNormal(truth.meanlog, truth.sdlog)

    fig = Figure(size = (1100, 200 + 60 * n + 40), figure_padding = (12, 18, 8, 8))
    Label(fig[1, 1:3], title;
          fontsize = 17, font = :bold, halign = :left, tellwidth = false)

    _render_outcome_row!(fig, 2, estimates, colors, truth_dist;
                         density_xlabel = "delay (days)")

    truth_el = LineElement(; color = :black, linestyle = :dash, linewidth = 3.5)
    method_els = [PolyElement(; color = (colors[i], 0.45), strokecolor = colors[i],
                              strokewidth = 1.2) for i in 1:n]
    n_total = n + 1
    nbanks = n_total ≤ 3 ? 1 : (n_total ≤ 6 ? 2 : 3)
    Legend(fig[3, 1:3],
        [truth_el; method_els],
        ["truth"; collect(labels)];
        orientation = :horizontal,
        nbanks = nbanks,
        framevisible = false,
        labelsize = 11,
        padding = (4, 4, 2, 2),
    )

    rowsize!(fig.layout, 1, Fixed(28))
    rowsize!(fig.layout, 3, Fixed(20 + 18 * nbanks))
    colsize!(fig.layout, 1, Relative(0.40))
    colsize!(fig.layout, 2, Relative(0.30))
    colsize!(fig.layout, 3, Relative(0.30))

    return fig
end

"""
    comparison_figure_two_outcomes(estimates_a, estimates_b, labels;
        truth_a, truth_b, title="",
        outcome_titles=("Outcome A", "Outcome B"),
        density_xlabels=("delay (days)", "delay (days)"))

Stack two `comparison_figure`-style rows in one PNG, one per outcome. Each row
gets its own shared x-axis across its median/mean panels. Posterior ridges are
drawn when the estimate NamedTuples carry `*_samples` fields, otherwise the
function falls back to dot + errorbar.
"""
function comparison_figure_two_outcomes(
    estimates_a::AbstractVector,
    estimates_b::AbstractVector,
    labels::AbstractVector;
    truth_a::NamedTuple,
    truth_b::NamedTuple,
    title::AbstractString = "",
    outcome_titles::Tuple{<:AbstractString, <:AbstractString} =
        ("Outcome A", "Outcome B"),
    density_xlabels::Tuple{<:AbstractString, <:AbstractString} =
        ("delay (days)", "delay (days)"),
)
    length(estimates_a) == length(estimates_b) == length(labels) ||
        error("estimates_a / estimates_b / labels length mismatch")
    n = length(labels)
    palette = Makie.wong_colors()
    colors = [palette[mod1(i, length(palette))] for i in 1:n]

    row_h = 60 * n + 40
    fig = Figure(size = (1100, 2 * row_h + 100), figure_padding = (12, 18, 8, 8))

    Label(fig[1, 1:3], title;
          fontsize = 17, font = :bold, halign = :left, tellwidth = false)
    Label(fig[2, 1:3], outcome_titles[1];
          fontsize = 13, font = :bold, halign = :left, tellwidth = false,
          color = (:black, 0.75))
    _render_outcome_row!(fig, 3, estimates_a, colors,
                         LogNormal(truth_a.meanlog, truth_a.sdlog);
                         density_xlabel = density_xlabels[1])

    Label(fig[4, 1:3], outcome_titles[2];
          fontsize = 13, font = :bold, halign = :left, tellwidth = false,
          color = (:black, 0.75))
    _render_outcome_row!(fig, 5, estimates_b, colors,
                         LogNormal(truth_b.meanlog, truth_b.sdlog);
                         density_xlabel = density_xlabels[2])

    truth_el = LineElement(; color = :black, linestyle = :dash, linewidth = 3.5)
    method_els = [PolyElement(; color = (colors[i], 0.45), strokecolor = colors[i],
                              strokewidth = 1.2) for i in 1:n]
    n_total = n + 1
    nbanks = n_total ≤ 3 ? 1 : (n_total ≤ 6 ? 2 : 3)
    Legend(fig[6, 1:3],
        [truth_el; method_els],
        ["truth"; collect(labels)];
        orientation = :horizontal,
        nbanks = nbanks,
        framevisible = false,
        labelsize = 11,
        padding = (4, 4, 2, 2),
    )

    rowsize!(fig.layout, 1, Fixed(28))
    rowsize!(fig.layout, 2, Fixed(22))
    rowsize!(fig.layout, 4, Fixed(22))
    rowsize!(fig.layout, 6, Fixed(20 + 18 * nbanks))
    colsize!(fig.layout, 1, Relative(0.40))
    colsize!(fig.layout, 2, Relative(0.30))
    colsize!(fig.layout, 3, Relative(0.30))

    return fig
end
