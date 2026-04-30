# Changelog

All notable changes to pydantic-cal are documented here. Format follows
[keep-a-changelog](https://keepachangelog.com/en/1.1.0/) 1.1.0;
project tracks [SemVer](https://semver.org/).

## [Unreleased]

### Added

- `[cache]` extra: pulls in `f3dx.cache.cached_call` so a calibration
  dataset is recorded once and replayed forever. CI runs offline
  via `F3DX_BENCH_OFFLINE=1` so a cache miss is a test failure
  instead of a silent live API hit.

## [0.0.3] - 2026-04-29

### Added

- smECE (Smooth ECE, Blasiok et al. 2023): kernel-smoothed
  Expected Calibration Error implementation. Less binning bias
  than vanilla ECE; the recommended default for fair model
  comparison across calibration sets of different sizes.

## [0.0.2] - 2026-04-28

### Added

- `pydantic-evals` adapter: per-case `Evaluator` and dataset-aggregate
  `ReportEvaluator` so users get
  `dataset.evaluate(task, report_evaluators=[ECE(), Brier(), ReliabilityDiagram()])`.
- Calibration metrics: ECE, MCE, ACE, Brier score,
  ReliabilityDiagram. Each matches the standard literature
  formulation; tests cover the edge cases (single-bucket,
  all-correct, all-wrong, perfect-calibration).

### Security

- SHA-pin every github action across CI.
- gitleaks, scorecard, dependabot enabled via org-level reusable
  workflows.

## [0.0.1] - earlier

Initial scaffold: package structure, CI baseline, OIDC Trusted
Publisher wired for PyPI release on tag push.
