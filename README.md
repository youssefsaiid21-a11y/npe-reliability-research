# NPE reliability under model misspecification

We want to find out how the authors' fixed NPE behaves when we change the model generating its input data.

**Now:** we can generate known parameters and time series for Random Walk, SEIRb and Generalized Bass on this Mac. We are waiting for the pretrained NPE files corresponding to our supplied manuscript. The older public GitHub files have not been verified as those networks. We will not train or retrain NPE.

## Start here

Read [what each file does](baseline-reproduction/START_HERE.md), or open our local visual lesson:

```sh
./thesis learn
```

The lesson uses actual generated examples. We can switch models, inspect a graph and reveal the known parameters behind it.

## Run the project

```sh
./thesis status
./thesis files
./thesis generate --model RW --cases 16
./thesis show --model RW
```

We can use `SEIRb` or `GB` instead of `RW`. Each generation command gets a new run folder. We keep existing runs.

## Our workflow

1. Choose known parameters and generate data with the authors' original model.
2. Give those observations to the authors' matching pretrained NPE.
3. Compare its estimates with the known parameters and the paper's results.
4. Once that baseline works, change one model assumption and repeat with the same fixed NPE.

Step 1 works for all three models. Steps 2–4 are waiting on the corresponding pretrained networks and baseline validation.

## References we can return to later

- [Simulation setup and provenance](baseline-reproduction/SIMULATION_SETUP.md)
- [Baseline checkpoints](baseline-reproduction/REPRODUCTION_PLAN.md)
- [Experimental design](experimental-design/EXPERIMENTAL_DESIGN.md)
- [Planning tool](experimental-design/planning.py)
- [Progress records](execution/2026-11-19-data-generation.md)

We save our code, notes and compact evidence to GitHub. Papers, private links, author archives/models, trained networks, environments and generated datasets remain local.
