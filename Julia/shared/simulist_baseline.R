#!/usr/bin/env Rscript
# Generate a "clean" line list with simulist that matches the baseline used by
# the OLD figures (R/generate-linelist.R in github.com/.../line-list-guide).
#
# The match covers:
#   - simulist::sim_linelist with the paper's three delay distributions and
#     outbreak_size = c(5000, 30000) (see the calibrated call below; the upper
#     bound ~ DDSA's N for comparable scale).
#   - hosp_risk: NOT passed (uses simulist's default), as in OLD. We previously
#     overrode this to 0.6 to keep onset-to-admission observable in most rows;
#     that override is dropped here for like-for-like comparison with the OLD
#     pipeline.
#   - 40% of recovered cases are flagged asymptomatic, with date_onset := NA
#     and (when admitted) date_reporting recentered around date_admission via
#     a rpois(lambda = 14) - 14 offset.
#   - Mild and severe symptom-onset columns: the simulated onset is treated
#     as severe; mild precedes severe by Poisson(lambda = 2) days. Cases
#     without a hospital admission have no severe onset.
#
# Post-hoc degradation (imputation, interval censoring, etc.) remains in the
# Julia downstream so each scenario can apply its own rule. This script just
# produces the clean baseline.
#
# Usage:
#   Rscript shared/simulist_baseline.R <output_csv> [seed]

suppressPackageStartupMessages({
  library(simulist)
  library(data.table)
})

args <- commandArgs(trailingOnly = TRUE)
out_csv <- if (length(args) >= 1) args[1] else "simulist_baseline.csv"
seed <- if (length(args) >= 2) as.integer(args[2]) else 1234L
set.seed(seed)

# Growth-rate calibration to match the DDSA SIR generator (R0 = 1.5,
# generation interval ~ 1/gamma = 2.5 days), so that *time-indexed* scenarios
# (real_time snapshots, reporting_scheme_change change-point) are comparable
# across the two pipelines. simulist's DEFAULTS produce a near-critical,
# slow-generation epidemic — contact mean 2 x prob_infection 0.5 = R0 ~ 1.0,
# and an infectious period averaging ~8.4 days — which sprawled over ~770 days
# and made absolute-day snapshots meaningless on this branch. We therefore set:
#   - contact_distribution mean 3 x prob_infection 0.5  => R0 ~ 1.5  (matches DDSA)
#   - infectious_period lognormal mean ~2.5 days        => generation interval
#                                                          ~ DDSA's 1/gamma
# simulist is a pure branching process (no susceptible depletion), so it grows
# exponentially until the outbreak_size cap rather than peaking-and-declining
# like the SIR; with these parameters the onset span is ~45-50 days, comparable
# to the DDSA growth phase. outbreak_size upper bound ~ DDSA's N for similar scale.
ll <- simulist::sim_linelist(
  contact_distribution = function(x) stats::dpois(x = x, lambda = 3),
  infectious_period    = function(x) stats::rlnorm(n = x, meanlog = 0.79,
                                                   sdlog = 0.5),
  prob_infection  = 0.5,
  outbreak_size   = c(5000, 30000),
  onset_to_hosp   = function(x) stats::rlnorm(n = x, meanlog = 1.5,
                                              sdlog = 0.5),
  onset_to_death  = function(x) stats::rlnorm(n = x, meanlog = 2.5,
                                              sdlog = 0.5),
  reporting_delay = function(x) stats::rgamma(n = x, shape = 3, scale = 1)
)

ll <- setDT(ll)

# Asymptomatic stratum (40% of recovered).
ll[, asymptomatic := FALSE]
ll[outcome == "recovered",
   asymptomatic := sample(c(TRUE, FALSE), size = .N, replace = TRUE,
                          prob = c(0.4, 0.6))]
ll[asymptomatic == TRUE, date_onset := NA]
ll[asymptomatic == TRUE & !is.na(date_admission),
   date_reporting := date_admission + rpois(.N, lambda = 14) - 14]

# Mild vs severe symptom onset: simulated onset is severe; mild precedes by
# Poisson(2). Non-admitted cases have no severe onset.
ll[!is.na(date_admission),
   date_onset_mild := date_onset - rpois(.N, lambda = 2)]
ll[!is.na(date_admission), date_onset_severe := date_onset]
ll[is.na(date_admission),  date_onset_mild   := date_onset]
ll[is.na(date_admission),  date_onset_severe := NA]

keep_cols <- c("id", "case_type", "outcome", "asymptomatic",
               "date_onset", "date_onset_mild", "date_onset_severe",
               "date_admission", "date_death",
               "date_first_contact", "date_last_contact",
               "date_reporting")
ll_keep <- ll[, intersect(keep_cols, names(ll)), with = FALSE]

write.csv(ll_keep, file = out_csv, row.names = FALSE, na = "")
cat("Wrote", nrow(ll_keep), "rows to", out_csv, "\n")
