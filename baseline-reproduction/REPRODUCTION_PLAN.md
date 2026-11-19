# Our baseline checkpoints

We use the authors' pretrained NPE. We do not train or retrain it.

## 1. Generate known parameters and data — working

We run the CPU generators supplied with the manuscript. The original model equations and source stay unchanged. We record the seed, settings and source fingerprints, then check each parameter/data pair.

Run `./thesis generate --model RW --cases 16` from the project root. The [file guide](START_HERE.md) explains the execution path.

## 2. Load the matching pretrained NPE — waiting for authors

Our supplied manuscript describes a public release after review. The older public GitHub repository is linked to earlier benchmarking work; we have not established that its saved networks produced this manuscript's results. The supplied replication archive contains no saved networks.

We need the corresponding trained networks, preprocessing, parameter order and evaluation settings. We will not mix versions or reshape observations silently to make an older estimator accept them.

## 3. Reproduce the original-model evaluation — pending

We apply the matching fixed NPE to generated observations, keeping the known parameters separate. We reproduce the manuscript's selection, preprocessing, parameter subsets, metrics and aggregation. We compare the resulting scores with the paper and account for simulation and posterior sampling uncertainty.

We define acceptable differences before marking this checkpoint as passed. Successful data generation alone is not a passed replication.

## 4. Introduce model misspecification — after the baseline passes

We change one declared mechanism at a time, generate new observations and use the same fixed NPE. We keep the evaluation consistent and explain which true parameter values remain meaningful after each change.

## Earlier diagnostics

We retained the older GitHub checkpoint checks and simulator-port pilots as separate evidence. They are not the current manuscript baseline. Their source is pinned to `d74a866f0476c07e93d786d297f90c0706281c33`; their limitations remain recorded.

We keep the benchmarking paper, appendix, extracts, private links, model assets and large generated files local. Only our own code, documentation and compact evidence go to GitHub.
