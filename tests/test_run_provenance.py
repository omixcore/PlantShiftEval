import argparse
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from run_provenance import complete_manifest, file_digest, split_record, write_manifest


class RunProvenanceTests(unittest.TestCase):
    def test_split_rejects_overlap_duplicate_empty_and_out_of_bounds(self):
        for train, validation, size in [([0], [0], 2), ([0, 0], [1], 2),
                                        ([], [1], 2), ([0], [], 2),
                                        ([-1], [1], 2), ([0], [2], 2)]:
            with self.subTest(train=train, validation=validation):
                with self.assertRaises(ValueError):
                    split_record(train, validation, size)

    def test_limited_split_keeps_original_cache_indices(self):
        record = split_record([9, 2], [5], 10)
        self.assertEqual(record['train_indices'], [9, 2])
        self.assertEqual(record['validation_indices'], [5])

    def test_manifest_identifies_inputs_and_only_completes_with_all_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cache, out, backgrounds = root / 'cache', root / 'out', root / 'bg'
            for folder in (cache, out, backgrounds):
                folder.mkdir()
            for name in ('meta.json', 'train_x.npy', 'train_y.npy', 'train_s.npy',
                         'test_x.npy', 'test_y.npy', 'test_s.npy'):
                (cache / name).write_bytes(name.encode())
            (backgrounds / 'x.npy').write_bytes(b'backgrounds')
            args = argparse.Namespace(cache=str(cache), out=str(out),
                                      bg_aug=True, bg_cache=str(backgrounds))
            path = write_manifest(args, split_record([0], [1], 2), ['train.py'])
            started = json.loads(path.read_text())
            self.assertEqual(started['status'], 'started')
            self.assertEqual(len(started['inputs']), 8)
            self.assertIn('transductive', started['target_access'])
            self.assertEqual(file_digest(backgrounds / 'x.npy'),
                             hashlib.sha256(b'backgrounds').hexdigest())
            with self.assertRaises(FileNotFoundError):
                complete_manifest(path)
            self.assertEqual(json.loads(path.read_text())['status'], 'started')
            for name in ('model.pt', 'metrics.json', 'logits.pt'):
                (out / name).write_bytes(name.encode())
            complete_manifest(path)
            completed = json.loads(path.read_text())
            self.assertEqual(completed['status'], 'completed')
            self.assertEqual(set(completed['outputs']), {'model.pt', 'metrics.json', 'logits.pt'})


if __name__ == '__main__':
    unittest.main()
