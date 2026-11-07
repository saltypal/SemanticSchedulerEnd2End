import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.spi_classifier_repair.minority_recovery import (
    PriorAdjustedForest, grouped_folds, load_development, choose)
from experiments.spi_classifier_repair.spi_label_audit import select_panel, analyze
from experiments.spi_classifier_repair.evidence_priority import parse_evidence


class MinorityRecoveryTests(unittest.TestCase):
    def test_evidence_protocol_rejects_unsupported_critical_and_preserves_unknown(self):
        row={"level":"Critical", "visible_evidence":["A person carries a weapon."],
             "immediate_harm_evidence":"not_established", "immediacy_cue":None, "uncertainty":None}
        with self.assertRaises(ValueError):
            parse_evidence(json.dumps(row))
        row.update(level=None, visible_evidence=[], uncertainty="Image is unreadable.")
        result,_=parse_evidence(json.dumps(row))
        self.assertFalse(result["available"])
        self.assertIsNone(result["level"])
        row.update(level="Medium",visible_evidence=["Checkout displays an unavailable-item error."],uncertainty=None)
        result,_=parse_evidence(json.dumps(row))
        self.assertEqual(result["level"],"Medium")

    def test_prior_adjustment_reference_and_tau_zero(self):
        class FakeForest:
            def predict_proba(self, x):
                return np.tile([.6, .1, .2, .1], (len(x), 1))
        model = PriorAdjustedForest(tau=1)
        model.model_, model.priors_, model.classes_ = FakeForest(), np.array([.6,.1,.2,.1]), np.arange(4)
        np.testing.assert_allclose(model.predict_proba(np.zeros((2, 1))), np.full((2,4), .25))
        model.tau = 0
        np.testing.assert_allclose(model.predict_proba(np.zeros((1, 1))), [[.6,.1,.2,.1]])

    def test_training_folds_group_disjoint_and_missing_class_rejected(self):
        y = np.repeat(np.arange(4), 18)
        groups = np.repeat(np.arange(36), 2)
        for train, val in grouped_folds(y, groups):
            self.assertFalse(set(groups[train]) & set(groups[val]))
            self.assertEqual(set(y[val]), set(range(4)))
        with self.assertRaises(ValueError):
            grouped_folds(np.repeat(np.arange(3), 18), np.arange(54))

    def test_old_test_poisoning_never_parsed_and_identity_permutation_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            rows = [{"image_id": str(i), "image_sha256": str(i), "group":str(i),
                     "split": ["train","validation","test"][i//4]} for i in range(12)]
            labels = [{**r, "available":True, "level":("Low","Medium","High","Critical")[i%4]}
                      for i,r in enumerate(rows)]
            np.savez(tmp/"features.npz", image_ids=[r["image_id"] for r in rows], embeddings=np.eye(12))
            (tmp/"records.json").write_text(json.dumps(rows))
            (tmp/"labels.json").write_text(json.dumps(labels))
            first = load_development(tmp/"features.npz",tmp/"records.json",tmp/"labels.json")
            for r in labels:
                if r["split"] == "test":
                    r["level"] = {"invalid":"must never be parsed"}
                    r["available"] = False
            (tmp/"labels.json").write_text(json.dumps(labels))
            second = load_development(tmp/"features.npz",tmp/"records.json",tmp/"labels.json")
            for role in first:
                np.testing.assert_array_equal(first[role]["y"], second[role]["y"])
                np.testing.assert_array_equal(first[role]["x"], second[role]["x"])
            rows[0]["image_id"]="foreign"
            (tmp/"records.json").write_text(json.dumps(rows))
            with self.assertRaises(ValueError):
                load_development(tmp/"features.npz",tmp/"records.json",tmp/"labels.json")

    def test_panel_is_training_only_and_disagreement_not_relabeling(self):
        levels=("Low","Medium","High","Critical")
        rows=[{"image_id":str(i),"split":"train" if i<8 else "test", "available":True,
               "level":levels[i%4]} for i in range(12)]
        selected=select_panel(rows,2)
        self.assertEqual(len(selected),8)
        self.assertTrue(all(r["split"]=="train" for r in selected))
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            (tmp/"withheld_key.json").write_text(json.dumps([{"case_id":"A01","image_id":"x","teacher_level":"Critical"}]))
            review={"review_type":"blinded_ai_diagnostic","cases":[{"case_id":"A01","plausible_levels":["High"],
                    "visible_evidence":"Damaged structure, no visible immediate threat.","immediate_harm_evidence":"not_established"}]}
            (tmp/"review.json").write_text(json.dumps(review))
            result=analyze(tmp,tmp/"review.json")
            self.assertFalse(result["automatic_relabeling"])
            self.assertFalse(result["independent_urgency_validated"])
            self.assertEqual(result["per_teacher_class"]["Critical"]["outside_reviewer_plausible_range"],1)


if __name__ == "__main__":
    unittest.main()
