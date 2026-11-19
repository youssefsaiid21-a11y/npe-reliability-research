# Read the files in this order

We are building a way to test the authors' fixed NPE on data from their models. **Right now, we can generate data. We are waiting for the manuscript's trained NPE files.**

## Follow one command

```sh
./thesis generate --model RW --cases 16
```

We can read that as: **run the Random Walk model 16 times and keep the known inputs alongside the generated observations.**

The command follows this path:

```text
thesis
  → thesis_cli.py
    → run_author_generator.py
      → the authors' synth_data_gen_pysdtorch.py
        → the authors' model equations
          → known parameters + time-series values
```

## What each file does

| File | Its job | What to read first |
|---|---|---|
| [thesis](../thesis) | Starts the tested Python environment. | The last line launches our command handler. |
| [thesis_cli.py](thesis_cli.py) | Reads commands such as `generate`, `show`, `status` and `learn`, then chooses the action. | `main()` |
| [run_author_generator.py](run_author_generator.py) | Checks the original source, runs the authors' generator, checks its output, and saves a receipt. | `main()`, then the functions it calls. |
| `synth_data_gen_pysdtorch.py` in the authors' folder | Chooses parameter values, runs simulations and applies its dataset selection rules. | We use the original generator; it is where the simulation work starts. |
| The model's `.mdl` file | Contains the equations: stocks, flows, constants and noise. | For Random Walk, start with `State` and `StateChng`. |
| `pysdtorch/` in the authors' folder | Executes those equations over time. | We can leave the engine internals until later. |

Our coordinator does not contain the epidemic or adoption equations. Those belong to the authors' model files.

## Read our coordinator as four steps

1. **`verify_source()`** checks that we are using the original files we recorded.
2. **`run_generator()`** launches the author's simulation script and keeps a log.
3. **`validate_outputs()`** checks that every time series has its matching parameters and sensible numeric values.
4. **`write_json()`** saves a receipt. A failed run gets a failure status, not a successful result.

The surrounding code handles command options and file locations. We do not need to understand every import before following these four steps.

## What gets saved?

For dataset 1:

- **`true_param_v01.cin`** is the answer sheet: the parameter values we used.
- **`data_v01.tab`** contains the generated numbers over time. We draw a graph from these numbers.
- **`generation-receipt.json`** records what ran and identifies the files.
- **`generator.log`** records progress and errors.

NPE will receive numerical observations with its required preprocessing. We keep the known parameters separately so we can check its estimates.

## The guide files only explain the work

- **[learning_guide.py](learning_guide.py)** reads a few checked examples and builds the local lesson.
- **[learning-guide.html](learning-guide.html)** controls the lesson's layout, diagrams and graph. It does not simulate anything.
- **[test_generation.py](test_generation.py)** checks that we reject broken parameter/data pairs and report failures correctly.
- **`environment-lock.txt`** lists the installed Python packages. It does not run an experiment.

Open the lesson with `./thesis learn`. Use `./thesis files` to print this file map in the terminal.

## Where do the older files fit?

The earlier `run_original_observation.py`, `checkpoint_runtime.py`, `prepare_source.py`, `generate_baseline_data.py` and `evaluate_baseline.py` belong to the older GitHub-network diagnostics. They are outside our current manuscript data-generation command. Their existence does not mean they run automatically.

The [technical setup reference](SIMULATION_SETUP.md) records versions and provenance. The experimental-design folder describes the later misspecification study. We can return to both after the active file path makes sense.

Our default `Thesis.code-workspace` shows the active files. `Thesis-reference.code-workspace` opens the full reference view. Both use the same project and files.

**Our next research step:** receive the authors' matching pretrained NPE, run the original-model evaluation, compare with the paper, and only then introduce model changes. We do not train NPE.
