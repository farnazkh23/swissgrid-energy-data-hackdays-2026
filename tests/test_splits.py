import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from swissgrid_forecaster.splits import Sample, RollingOrigin

T = datetime(2026, 1, 1, tzinfo=timezone.utc)
H = timedelta(hours=1)


def samples(n=50):
    return tuple(Sample(str(i), T+i*H, T+(i+.5)*H, T+i*H,
                        T+(i+1)*H, (float(i % 2),), float(i)) for i in range(n))


def plan():
    return RollingOrigin(T+10*H, T+40*H, 10*H, 4*H, 2*H, 10*H, H, H, H)


class SplitTests(unittest.TestCase):
    def test_deterministic_folds(self):
        p = plan()
        self.assertEqual(p.split(samples()), p.split(reversed(samples())))
        self.assertEqual(len(p.split(samples())), 3)
        self.assertNotEqual(p.split(samples())[0].fold_id, replace(p, purge=2*H).split(samples())[0].fold_id)

    def test_purge_embargo_and_label_cutoff(self):
        f = plan().split(samples())[0]
        self.assertEqual(f.fit_cutoff, T+8*H)
        self.assertTrue(all(r.label_known_at <= f.fit_cutoff for r in f.train))
        self.assertEqual([r.row_id for r in f.train], list(map(str, range(8))))
        self.assertEqual([r.row_id for r in f.validation], ['10', '11', '12', '13'])
        self.assertEqual([r.row_id for r in f.calibration], ['15', '16'])
        ids = [{r.row_id for r in part} for part in (f.train, f.validation, f.calibration)]
        self.assertFalse(ids[0] & ids[1] or ids[1] & ids[2])

    def test_holdout_isolation(self):
        original = samples()
        changed = tuple(replace(r, target=999999) if r.issue_time >= plan().holdout_start else r for r in original)
        self.assertEqual(plan().split(original), plan().split(changed))
        self.assertEqual(len(plan().holdout(original)), 10)
        crossing = replace(original[38], target_time=T+41*H, label_known_at=T+42*H)
        for f in plan().split((*original[:38], crossing, *original[39:])):
            self.assertNotIn(crossing.row_id, [r.row_id for r in (*f.train, *f.validation, *f.calibration)])

    def test_late_labels_excluded(self):
        rows = samples()
        rows = (replace(rows[0], label_known_at=T+30*H), *rows[1:])
        self.assertNotIn('0', [r.row_id for r in plan().split(rows)[0].train])

    def test_embargo_delays_next_origin_even_with_short_step(self):
        p = replace(plan(), step=H)
        folds = p.split(samples())
        # First calibration ends at hour 17; next train cutoff cannot
        # precede hour 18, after the full one-hour embargo.
        self.assertEqual(folds[1].fit_cutoff, T+18*H)
        self.assertEqual(folds[1].validation[0].issue_time, T+20*H)

    def test_invalid_inputs(self):
        for change in [dict(step=timedelta(0)), dict(purge=-H), dict(first_origin=T.replace(tzinfo=None))]:
            with self.assertRaises(ValueError):
                replace(plan(), **change)
        with self.assertRaises(ValueError):
            plan().split((samples()[0], samples()[0]))
