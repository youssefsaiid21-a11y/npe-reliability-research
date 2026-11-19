"""Protect the experiment's pairing checks and failure reporting."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_author_generator as generator


class GenerationChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.series = self.root / 'true_data/true_series_tab'
        self.truths = self.root / 'true_data/true_params'
        self.series.mkdir(parents=True)
        self.truths.mkdir(parents=True)
        for index in [1, 2]:
            (self.series / f'data_v{index:02d}.tab').write_text('Time\tMN1 Data\n0\t4\n1\t4.2\n')
            (self.truths / f'true_param_v{index:02d}.cin').write_text('Drift = 0.2\nNSeed = 31\n')

    def tearDown(self):
        self.temporary.cleanup()

    def check(self):
        return generator.validate_outputs(self.root, 2, {'Drift': (-1, 1)})

    def test_correct_pairs_have_fingerprints(self):
        result = self.check()
        self.assertEqual(result['rows_per_dataset'], 2)
        self.assertEqual(len(result['records']), 2)
        self.assertEqual(len(result['records'][0]['series_sha256']), 64)

    def test_same_counts_with_wrong_pairing_are_rejected(self):
        (self.truths / 'true_param_v02.cin').rename(self.truths / 'true_param_v03.cin')
        with self.assertRaisesRegex(RuntimeError, 'matching parameter'):
            self.check()

    def test_out_of_bounds_truth_is_rejected(self):
        (self.truths / 'true_param_v02.cin').write_text('Drift = 12\n')
        with self.assertRaisesRegex(RuntimeError, 'out-of-bounds'):
            self.check()

    def test_misaligned_times_are_rejected(self):
        (self.series / 'data_v02.tab').write_text('Time\tMN1 Data\n0\t4\n2\t4.2\n')
        with self.assertRaisesRegex(RuntimeError, 'inconsistent'):
            self.check()

    def test_misaligned_series_names_are_rejected(self):
        (self.series / 'data_v02.tab').write_text('Time\tMN2 Data\n0\t4\n1\t4.2\n')
        with self.assertRaisesRegex(RuntimeError, 'inconsistent'):
            self.check()

    def test_nonfinite_observation_is_rejected(self):
        (self.series / 'data_v02.tab').write_text('Time\tMN1 Data\n0\t4\n1\tinf\n')
        with self.assertRaisesRegex(RuntimeError, 'Non-finite'):
            self.check()

    def test_repeated_parameter_name_is_rejected(self):
        (self.truths / 'true_param_v02.cin').write_text('Drift = 0.2\nDrift = 0.3\n')
        with self.assertRaisesRegex(RuntimeError, 'Duplicate'):
            self.check()

    def test_lesson_reads_the_checked_graph_and_answer_pair(self):
        from learning_guide import read_example
        generator.write_json(self.root / 'generation-receipt.json', self.check())
        example = read_example(self.root, 1)
        self.assertEqual(example['times'], [0, 1])
        self.assertEqual(example['series']['MN1 Data'], [4, 4.2])
        self.assertEqual(example['parameters'][0]['value'], .2)

    def test_lesson_rejects_an_answer_changed_after_validation(self):
        from learning_guide import read_example
        generator.write_json(self.root / 'generation-receipt.json', self.check())
        (self.truths / 'true_param_v01.cin').write_text('Drift = 0.3\n')
        with self.assertRaisesRegex(ValueError, 'changed since validation'):
            read_example(self.root, 1)

    def test_generator_failure_never_creates_a_success_receipt(self):
        script = self.root / 'author/synth_data_gen_pysdtorch.py'
        script.parent.mkdir()
        (script.parent / 'Synth_Data_Gen.voc').write_text('-1<=Drift<=1\n')
        output = self.root / 'results/failure'
        with patch.object(generator, 'RUNTIME', self.root), \
             patch.object(generator, 'verify_source', return_value=(script, 1)), \
             patch.object(generator, 'run_generator', side_effect=RuntimeError('Simulation failed')):
            with self.assertRaisesRegex(RuntimeError, 'Simulation failed'):
                generator.main(['--model', 'RW', '--cases', '2', '--output', str(output)])
        self.assertFalse((output / 'generation-receipt.json').exists())
        self.assertEqual(json.loads((output / 'run-status.json').read_text())['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
