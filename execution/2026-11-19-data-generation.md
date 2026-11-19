# Data generation environment — 19 November 2026

The manuscript authors' own CPU generators now run on this ARM64 Mac. No NPE training or retraining was executed.

- Generated 500 Random Walk datasets, 16 SEIRb datasets and 16 Generalized Bass datasets, each paired with known parameters. Author screening/stratification defaults were retained.
- Verified finite outputs, paired file counts, parameter bounds and observation time axes. Repeated an eight-case RW run: all 16 parameter/data files were byte-identical.
- Rechecked the wrapper with two additional cases for SEIRb and GB. The SEIRb C++ backend works locally; RW and GB used the Torch CPU backend. One worker, at most two numerical threads.
- Source replication archive SHA-256: `fdfbe324b0dc260651780ed5b9e813b85ac78f35f36a234002cc7ee9aba54f9d`. Executed model source files were compared against archive contents. No private download link is published.
- Verified all 29 required pinned GitHub artifacts across the three models. `uv pip check` passed for the 136-package environment; lockfile updated.
- Before locating the author CPU route, separate small GitHub-package/PySD pilots ran through fixed 100k NPE checkpoints. Those are engine-port diagnostics, not manuscript replication. The optional GB pink-noise expansion remains provisional. Their compact evidence is retained separately.

## Readiness decision

Data generation: working. Manuscript NPE baseline: **not passed**. The manuscript archive contains no saved networks. The publicly available GitHub estimators have differing model configurations, including RW process noise/time axis and GB parameter dimensions/ranges. The paper also scores structural parameters separately from noise parameters. A direct numerical match cannot be certified by mixing these releases.

The next external input is the pretrained weights corresponding to the manuscript models, plus matching preprocessing, parameter ordering and posterior construction. Do not train replacements or silently reshape/relabel manuscript data to fit older weights.

Vensim is not required for the successful CPU generation route. A university license could still help native-engine validation; it does not resolve checkpoint provenance.

Our code, documentation and compact receipts are suitable for GitHub. Papers, extracted text, private links, downloaded archive/source, model binaries, checkpoints, generated datasets and posterior arrays remain local.
