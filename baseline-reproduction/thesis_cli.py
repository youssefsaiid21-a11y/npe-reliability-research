"""Our front door: generate data, inspect an example, or open our lesson.

We keep command handling here. The model equations and output checks belong
to other files, so we can read the command flow without reading the simulator.
"""
import argparse
import json
import re
import sys
import webbrowser
from datetime import datetime, timezone
from pathlib import Path

from run_author_generator import MODELS, RUNTIME, main as generate, write_json

DEFAULT_RUNS = {'RW': 'author-RW-500', 'SEIRb': 'author-SEIRb-16', 'GB': 'author-GB-16'}
LATEST = RUNTIME / 'results/latest-runs.json'


def available_runs():
    """We show our latest completed runs, with the first verified runs as defaults."""
    runs = {code: RUNTIME / 'results' / folder for code, folder in DEFAULT_RUNS.items()}
    if LATEST.exists():
        for code, folder in json.loads(LATEST.read_text()).items():
            path = Path(folder).resolve()
            if code in MODELS and path.is_relative_to((RUNTIME / 'results').resolve()):
                runs[code] = path
    return runs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('status', help='Show what works and what is waiting')
    commands.add_parser('files', help='Explain the files in the order they run')
    lesson = commands.add_parser('learn', help='Open a visual explanation with real generated examples')
    lesson.add_argument('--no-open', action='store_true', help='Build the page without opening a browser')
    run = commands.add_parser('generate', help='Generate time series and their known parameters')
    run.add_argument('--model', choices=list(MODELS), default='RW')
    run.add_argument('--cases', type=int, default=16)
    run.add_argument('--seed', type=int, default=31)
    run.add_argument('--name', help='New run folder name; automatic if omitted')
    show = commands.add_parser('show', help='Read one generated example and its known parameters')
    show.add_argument('--model', choices=list(MODELS), default='RW')
    show.add_argument('--case', type=int, default=1)
    args = parser.parse_args(argv)
    try:
        if args.command == 'generate':
            # We make a unique folder name unless we choose one explicitly.
            name = args.name or args.model + '-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
            if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', name):
                parser.error('--name can contain letters, digits, underscores and hyphens')
            output = generate(['--model', args.model, '--cases', str(args.cases), '--seed', str(args.seed),
                               '--output', str(RUNTIME / 'results/runs' / name)])
            # We update this pointer only after generation and checks succeed.
            latest = json.loads(LATEST.read_text()) if LATEST.exists() else {}
            latest[args.model] = str(output)
            write_json(LATEST, latest)
            print('\nTo inspect an example: ./thesis show --model ' + args.model)
        elif args.command == 'status':
            print('DATA GENERATION: ready. Uses the authors’ own CPU simulators.')
            for code, path in available_runs().items():
                receipt = path / 'generation-receipt.json'
                if receipt.exists():
                    data = json.loads(receipt.read_text())
                    print(f'  {MODELS[code]["name"]}: {data["cases"]} datasets in {path.name}')
                else:
                    print(f'  {MODELS[code]["name"]}: no completed run at {path.name}')
            print('\nPAPER NPE: waiting for the authors’ corresponding pretrained networks.')
            print('PAPER COMPARISON: pending. No NPE training is performed.')
        elif args.command == 'files':
            print('Follow ./thesis generate through these files:\n')
            for name, role in [
                ('thesis', 'Starts our tested Python environment.'),
                ('thesis_cli.py', 'Chooses the action from our command.'),
                ('run_author_generator.py', 'Checks source → runs the generator → checks outputs → saves a receipt.'),
                ('authors: synth_data_gen_pysdtorch.py', 'Chooses parameters and runs the model.'),
                ('authors: model .mdl file', 'Contains the stocks, flows, constants and noise equations.'),
                ('outputs: .cin + .tab', 'Known parameters paired with the generated time-series values.'),
            ]:
                print(name + '\n  ' + role + '\n')
            print('We can start with main() in run_author_generator.py. Details: baseline-reproduction/START_HERE.md')
        elif args.command == 'show':
            from learning_guide import read_example
            sample = read_example(available_runs()[args.model], args.case)
            print(f'{MODELS[args.model]["name"]}, dataset {args.case}: known simulation inputs')
            for parameter in sample['parameters']:
                print(f'  {parameter["name"]}: {parameter["value"]:g}')
            print(f'\nGenerated {len(sample["times"])} observation times. These are not NPE estimates.')
            print('See the actual graph: ./thesis learn')
        else:
            from learning_guide import build_guide
            page = build_guide(available_runs())
            print(page)
            if not args.no_open:
                webbrowser.open(page.as_uri())
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print('Could not complete this command: ' + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
