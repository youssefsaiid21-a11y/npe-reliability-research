"""We build our local lesson from checked examples, without running simulations.

The HTML template is shared code. We put the filled page and its real data in
our ignored runtime folder, so our raw examples stay local.
"""
import csv
import hashlib
import json
import re
from pathlib import Path

from run_author_generator import BASE, MODELS, RUNTIME

STORIES = {'RW': 'A quantity wanders up and down over time.',
           'SEIRb': 'An epidemic spreads while people change their behavior.',
           'GB': 'A product gains adopters through innovation and imitation.'}
MEANINGS = {'Drift': 'Tendency for the state to rise or fall.',
            'InitS': 'The starting state.',
            'SM1': 'Measurement noise in the first observed series.',
            'SM2': 'Measurement noise in the second observed series.',
            'SM3': 'Measurement noise in the third observed series.',
            'sp1': 'Size of the process noise inside the model.',
            'b1': 'Sensitivity to price.', 'b2': 'Sensitivity to advertising.',
            'Imitation q': 'Influence of existing adopters on new adoption.',
            'Initial Adopters': 'How many people have adopted at the start.',
            'Innovation p': 'Adoption independent of existing adopters.',
            'IFR': 'Fraction of infections that lead to death.',
            'Patient Zero Arrival Time': 'When the first infected person arrives.',
            'PWRisk Gamma': 'Shape of the behavioral response to perceived risk.',
            'Sensitivity to death Alpha': 'Strength of the response to perceived deaths.',
            'Time to perceive Tp': 'How quickly rising risk is perceived.',
            'Time to reduce risk': 'How quickly perceived risk falls.',
            'Transmission intensity Beta': 'Baseline intensity of transmission.'}


def read_example(run, case):
    """Read the matching graph and answer file, checking the saved fingerprints."""
    receipt = json.loads((run / 'generation-receipt.json').read_text())
    records = [r for r in receipt['records'] if re.fullmatch(r'data_v0*' + str(case) + r'\.tab', Path(r['series']).name)]
    if len(records) != 1:
        raise ValueError('That dataset is not present in the selected completed run')
    record = records[0]
    paths = {}
    for key in ['series', 'parameters']:
        # We check both halves of the example before showing them together.
        path = (run / record[key]).resolve()
        if not path.is_relative_to(run.resolve()):
            raise ValueError('Invalid dataset path')
        if hashlib.sha256(path.read_bytes()).hexdigest() != record[key + '_sha256']:
            raise ValueError('Example changed since validation: ' + path.name)
        paths[key] = path
    with paths['series'].open() as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    columns = [key for key in rows[0] if re.fullmatch(r'MN\d+ Data', key)]
    if not columns:
        raise ValueError('No observed model series found')
    parameters = []
    for line in paths['parameters'].read_text().splitlines():
        name, value = line.split('=', 1)
        name = name.strip()
        if name != 'NSeed':
            parameters.append(dict(name=name, value=float(value), meaning=MEANINGS.get(name, 'A model input.')))
    return dict(id=case, times=[float(r['Time']) for r in rows],
                series={key: [float(row[key]) for row in rows] for key in columns}, parameters=parameters)


def build_guide(runs):
    # Three examples per model keep our lesson small. This is a display choice,
    # not a statistical selection rule for the eventual NPE evaluation.
    models = []
    for code, run in runs.items():
        if not (run / 'generation-receipt.json').exists():
            continue
        receipt = json.loads((run / 'generation-receipt.json').read_text())
        models.append(dict(code=code, name=MODELS[code]['name'], story=STORIES[code],
                           cases=receipt['cases'], examples=[read_example(run, case) for case in range(1, min(3, receipt['cases']) + 1)]))
    text = (BASE / 'learning-guide.html').read_text()
    payload = json.dumps(dict(models=models), allow_nan=False).replace('<', '\\u003c')
    text, count = re.subn(r'(<script id="examples" type="application/json">).*?(</script>)',
                         lambda m: m[1] + payload + m[2], text, flags=re.S)
    if count != 1:
        raise RuntimeError('Lesson data placeholder is missing')
    output = RUNTIME / 'learning/start-here.html'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    return output
