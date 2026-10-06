# ROVA-Evaluation v2.0 — release notes

Revision release of the evaluation artifact for *"The Verdict Is in the Protocol: A
Circularity-Controlled, Cost-Ratio-Resolved Evaluation of Runtime Governance for Agentic AI"*
(IEEE Access, under review, 2026). It adds everything the revised paper's Appendix D cites
beyond v1.0. Nothing from v1.0 is removed, and no archived output or tagged commit is
modified.

## Added

* **Original internal protocol** — `docs/PRE_REGISTERED_PROTOCOL.md`, published verbatim
  (SHA-256 `7d87b08f1e15bdcea9997918d75658718ce3316708f45507816ee19e3af23712`). It was never
  registered externally.
* **Deviation record** — `docs/PREREGISTRATION_DEVIATIONS.md`: each commitment, what the
  original study executed, how the revision restores it, the matched-world correction, and
  the referenced documents that were not retained.
* **Dated analysis plans** — `docs/REVISION_ANALYSIS_PLAN.md` (REV-001) and
  `docs/REVISION_002_ANALYSIS_PLAN.md` (REV-002), each written after the previous results
  were known and before its own execution; and `docs/REVISION_002_POSTRUN_ANALYSES.md`
  (exploratory P1).
* **Revision code** — `rova_revision/`, with tests (8 modules).
* **Revision outputs** — `experiments/REV-001_revision_analyses/`,
  `experiments/REV-002_A1_tuning_convergence/`, `experiments/REV-002_A2_cluster_reanalysis/`,
  `experiments/REV-002_P1_behaviour_a0/`, indexed in `experiments/REVISION_OUTPUTS.md`.

## Tags cited in the paper (Appendix D.1)

| Tag | Commit |
|---|---|
| `main-001-frozen` | `7f624ca27a86fbbbe67afc68ebb886a4edbb67a2` (in v1.0) |
| `ablation-001-frozen` | `4533c61aa2e252698fd0c31d5bc1f9b9d04c18f0` (in v1.0) |
| `revision-001-analysis-plan` | `8dbb054d4586e44fddd85435feaa760b4b6a9e97` |
| `revision-001-frozen` | `053a1fa84b91644338f025154b3c4aa42da9fe9d` |
| `revision-002-analysis-plan` | `58affad5d44826d03720f30ddc72e37e6e2bb3d3` |
| `revision-002-frozen` | `226186c29c9a4a151f14ea12b4fd8e6a4bdd4032` |

## Output archives

| Archive | SHA-256 |
|---|---|
| `revision_out.tar.gz` (REV-001) | `63dae057d63c4917d90391fd9da72279f5e909ffb1a1b7e192999f55d28a6b10` |
| `a1_out.tar.gz` (A1) | `35459e1e096135531849a3a96d3f7501392c8d4f10474f00bc83913baa908e03` |

Both are in the repository under `experiments/`; the copies attached to this release are
identical.

## Notes

* **Line endings.** `.gitattributes` now checks out every file byte for byte. With
  `core.autocrlf` enabled (the Git for Windows default), v1.0 checkouts converted text files to
  CRLF and the MAIN-001 and ABLATION-001 checksum manifests failed; they now verify on every
  platform. No committed file changed.
* **v1.0 documents.** The experiment reports, the v1.0 entries of `docs/EXPERIMENT_INDEX.md`
  and `docs/CHANGELOG.md`, `docs/HOSTILE_SELF_REVIEW.md` and `docs/RELEASE_NOTES_v1.0.md` are
  kept as written. Where the revision analyses reach a different conclusion, those documents
  are superseded.
