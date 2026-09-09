from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from swissgrid_forecaster.manifest import ManifestEntry
from swissgrid_forecaster.raw_store import IntegrityError, RawStore

T = datetime(2026, 1, 1, tzinfo=timezone.utc)


class RawStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = RawStore(self.temp.name)
        self.meta = dict(source_id='provider', source_record_id='record', revision_id='v1',
                         first_received_at=T, source_metadata={'endpoint': 'fixture'})

    def test_exact_hash_and_manifest_roundtrip(self):
        payload = b'{ "b": 2, "a": 1 }\n'
        with patch('swissgrid_forecaster.raw_store.sha256', wraps=sha256) as hasher:
            result = self.store.put(payload, **self.meta)
            self.assertEqual(hasher.call_count, 1)
        entry = result.manifest
        self.assertEqual(entry.sha256, sha256(payload).hexdigest())
        self.assertEqual(self.store.read_receipt(entry), payload)
        self.assertEqual(ManifestEntry.from_bytes(entry.to_bytes()), entry)
        self.assertEqual(self.store.manifests(), (entry,))

    def test_duplicates_preserve_receipts_not_duplicate_bytes(self):
        first = self.store.put(b'raw', **self.meta)
        second = self.store.put(b'raw', **{**self.meta, 'revision_id': 'v2', 'source_id': 'other'})
        self.assertFalse(first.duplicate_payload)
        self.assertTrue(second.duplicate_payload)
        self.assertEqual(len(list(self.store.raw_dir.iterdir())), 1)
        self.assertEqual(len(self.store.manifests()), 2)
        self.assertEqual(first.manifest.sha256, second.manifest.sha256)

    def test_corruption_rejected_without_overwrite(self):
        entry = self.store.put(b'raw', **self.meta).manifest
        path = self.store.raw_dir / entry.sha256
        path.chmod(0o644)
        path.write_bytes(b'bad')
        with self.assertRaises(IntegrityError):
            self.store.read(entry.sha256)
        with self.assertRaises(IntegrityError):
            self.store.put(b'raw', **self.meta)
        self.assertEqual(path.read_bytes(), b'bad')
        self.assertEqual(len(self.store.manifests()), 1)

    def test_completed_retrieval_before_clock(self):
        events = []
        def fetch():
            events.append('completed')
            return b'raw'
        def clock():
            self.assertEqual(events, ['completed'])
            events.append('clock')
            return T
        result = self.store.retrieve(fetch, clock=clock, source_id='p',
                                     source_record_id='r', revision_id='v')
        self.assertEqual(result.manifest.first_received_at, T)

    def test_failed_retrieval_has_no_timestamp_or_manifest(self):
        def fail():
            raise OSError('retrieval failed')
        with patch('swissgrid_forecaster.raw_store.datetime') as clock:
            with self.assertRaises(OSError):
                self.store.retrieve(fail, source_id='p', source_record_id='r', revision_id='v')
            clock.now.assert_not_called()
        self.assertEqual(self.store.manifests(), ())

    def test_concurrent_duplicate_publication(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: self.store.put(b'same', **self.meta), range(12)))
        self.assertEqual(sum(not r.duplicate_payload for r in results), 1)
        self.assertEqual(len(list(self.store.raw_dir.iterdir())), 1)
        self.assertEqual(len(self.store.manifests()), 12)
        self.assertEqual(self.store.read(results[0].manifest.sha256), b'same')

    def test_failed_atomic_link_leaves_no_partial_file(self):
        with patch('swissgrid_forecaster.raw_store.os.link', side_effect=OSError('failure')):
            with self.assertRaises(OSError):
                self.store.put(b'raw', **self.meta)
        self.assertEqual(list(self.store.raw_dir.iterdir()), [])
        self.assertEqual(self.store.manifests(), ())

    def test_invalid_receipt_and_digest(self):
        with self.assertRaises(ValueError):
            self.store.put(b'raw', **{**self.meta, 'first_received_at': T.replace(tzinfo=None)})
        self.assertEqual(list(self.store.raw_dir.iterdir()), [])
        with self.assertRaises(ValueError):
            self.store.read('../escape')
        entry = self.store.put(b'raw', **self.meta).manifest
        with self.assertRaises(IntegrityError):
            self.store.read_receipt(replace(entry, byte_count=99))
