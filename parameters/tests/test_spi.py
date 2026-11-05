"""Engineering fixtures only: no real urgency performance is inferred here."""
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np

from parameters.contracts import MissionSpec
from parameters.extractor import VisualEvidence, SenderMemory
from parameters.minimal_parameters import MinimalHeader, MinimalParameterExtractor
from parameters.pilot_parameters import context_similarity_key
from parameters.spi import metrics, prepare, train, TrainingConfig, SPIClassifier, validate_levels
from parameters.spi_labels import parse_label, LEVELS, label_image
from parameters.spi_campaign import execute, verify_inputs
import hashlib
import zipfile


def fixture(root):
    rng=np.random.default_rng(9); vectors=[]; rows=[]
    for split,count in (("train",5),("validation",2),("test",1)):
        for level in range(4):
            for k in range(count):
                identity=f"{split}-{level}-{k}"; vector=np.eye(4)[level]+rng.normal(0,.01,4)
                vectors.append(vector)
                rows.append({"image_id":identity,"image_sha256":identity,"group":identity,"split":split,
                    "available":True,"level":LEVELS[level],"label_source":"vlm_pseudo",
                    "label_unit":"whole_image","rubric_sha256":"engineering_fixture"})
    np.savez_compressed(root/"features.npz",image_ids=np.array([r["image_id"] for r in rows]),
        image_sha256=np.array([r["image_sha256"] for r in rows]),embeddings=np.array(vectors,np.float32),
        feature_signature=np.array("fixture_features"))
    (root/"labels.json").write_text(json.dumps(rows),encoding="utf8")
    return rows


class SPITests(unittest.TestCase):
    def test_hand_computed_ordinal_score(self):
        self.assertEqual(metrics(np.array([[1.,0,0,0]]),np.array([0]))["rps"],0)
        self.assertEqual(metrics(np.array([[0.,0,0,1]]),np.array([0]))["rps"],1)
        self.assertAlmostEqual(metrics(np.array([[0.,1,0,0]]),np.array([0]))["rps"],1/3)

    def test_parser_abstains_instead_of_forcing_low(self):
        valid={"available":False,"level":None,"evidence":"blurry","reason":"not enough visual evidence"}
        self.assertFalse(parse_label(json.dumps(valid))["available"])
        for change in ({"level":"Low"},{"available":"false"},{"reason":None}):
            with self.assertRaises(ValueError):parse_label(json.dumps({**valid,**change}))

    def test_available_is_derived_without_changing_or_inventing_level(self):
        raw={"level":"High","evidence":"concrete hazard","reason":None}
        self.assertEqual(parse_label(json.dumps(raw)),{**raw,"available":True})
        self.assertEqual(parse_label(json.dumps({**raw,"reason":"null"})),{**raw,"available":True})
        self.assertFalse(parse_label('{"level":null,"evidence":"unclear","reason":"ambiguous"}')["available"])
        for change in ({"level":"EMERGENCY"},{"available":False},{"reason":"contradictory"}):
            with self.assertRaises(ValueError):parse_label(json.dumps({**raw,**change}))

    def test_teacher_adapter_uses_shared_tuple_parser_without_truth_inputs(self):
        class Reviewer:
            provenance={"model":"engineering_fixture"}
            def review_structured(self,paths,payload,prompt,parser,case_id,output,*,protocol):
                self.payload=payload
                parsed,recovered=parser('{"available":true,"level":"High","evidence":"visible blocked route","reason":null}')
                return {"parsed":parsed,"parse_recovery_used":recovered,"status":"complete"}
        reviewer=Reviewer()
        result=label_image(reviewer,"fixture.png","image01","unused_fixture_path")
        self.assertEqual(result["level"],"High")
        self.assertEqual(set(reviewer.payload),{"instruction"})
        self.assertEqual(result["label_source"],"vlm_pseudo")
        self.assertEqual(result["teacher"]["mode"],"fixed_rubric_image_only_spi_label")
        self.assertEqual(result["teacher"]["system_prompt_sha256"],result["prompt_sha256"])

    def test_campaign_preflight_failure_saves_status_and_closes_teacher(self):
        class Reviewer:
            closed=False
            def close(self):self.closed=True
        with TemporaryDirectory() as directory:
            reviewer=Reviewer();p=Path(directory)
            with self.assertRaises(FileNotFoundError):execute(p/"missing",p/"run",reviewer)
            status=json.loads((p/"run/status.json").read_text())
            self.assertEqual(status["status"],"incomplete");self.assertTrue(reviewer.closed)

    def test_zip_and_kaggle_expansion_preserve_exact_image_identity(self):
        with TemporaryDirectory() as directory:
            p=Path(directory);data=b"engineering image bytes";sha=hashlib.sha256(data).hexdigest()
            rows=[{"image_path":"images/one.png","image_sha256":sha}]
            (p/"panel.json").write_text(json.dumps(rows));(p/"features.npz").write_bytes(b"fixture features")
            with zipfile.ZipFile(p/"images.zip","w") as archive:archive.writestr("images/one.png",data)
            manifest={"files":{n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in ("panel.json","features.npz","images.zip")}}
            (p/"input_manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(verify_inputs(p)[1]["layout"],"zip")
            (p/"images/images").mkdir(parents=True);(p/"images/images/one.png").write_bytes(data)
            (p/"images.zip").unlink()
            self.assertEqual(verify_inputs(p)[1]["layout"],"kaggle_expanded_zip")
            (p/"images/images/one.png").write_bytes(b"corrupt")
            with self.assertRaises(ValueError):verify_inputs(p)

    def test_bad_source_or_leaking_group_is_rejected(self):
        with TemporaryDirectory() as directory:
            p=Path(directory);rows=fixture(p)
            rows[0]["label_source"]="message_priority"
            (p/"labels.json").write_text(json.dumps(rows))
            with self.assertRaises(ValueError):prepare(p/"features.npz",p/"labels.json")
            rows=fixture(p);rows[-1]["group"]=rows[0]["group"]
            (p/"labels.json").write_text(json.dumps(rows))
            with self.assertRaises(ValueError):prepare(p/"features.npz",p/"labels.json")

    def test_missing_class_is_not_fabricated(self):
        with TemporaryDirectory() as directory:
            p=Path(directory);rows=fixture(p)
            for r in rows:
                if r["split"]=="train" and r["level"]=="Critical":r.update(available=False,level=None,reason="teacher abstained")
            (p/"labels.json").write_text(json.dumps(rows))
            with self.assertRaises(ValueError):prepare(p/"features.npz",p/"labels.json")

    def test_training_export_and_heldout_poisoning(self):
        with TemporaryDirectory() as directory:
            p=Path(directory);rows=fixture(p)
            config=TrainingConfig(epochs=10,hidden_dimension=12,learning_rate=.02,patience=10)
            model,report=train(p/"features.npz",p/"labels.json",p/"run1",config)
            self.assertFalse(report["test_evaluated"])
            self.assertLess(report["export_max_absolute_difference"],1e-6)
            self.assertGreater(report["student"]["accuracy"],report["training_prior"]["accuracy"])
            for r in rows:
                if r["split"]=="test":r.update(level="POISONED",available="POISONED")
            (p/"labels.json").write_text(json.dumps(rows))
            model2,_=train(p/"features.npz",p/"labels.json",p/"run2",config)
            self.assertEqual(model.artifact["sha256"],model2.artifact["sha256"])
            with self.assertRaises(ValueError):model.predict(np.ones(4),"wrong_encoder")
            corrupt={**model.artifact,"b2":[0.,0.,0.,0.]}
            with self.assertRaises(ValueError):SPIClassifier(corrupt)

    def test_explicit_three_class_training_and_canonical_header_mapping(self):
        with TemporaryDirectory() as directory:
            p=Path(directory);rows=fixture(p)
            levels=("Low","High","Critical")
            # Removing labelled Medium examples requires an explicitly different
            # cohort in this engineering fixture, never a production fallback.
            kept=[r for r in rows if r["level"]!="Medium"]
            (p/"labels.json").write_text(json.dumps(kept))
            with self.assertRaises(ValueError):prepare(p/"features.npz",p/"labels.json")
            model,report=train(p/"features.npz",p/"labels.json",p/"run",
                TrainingConfig(levels=levels,epochs=8,hidden_dimension=12,learning_rate=.02))
            self.assertEqual(model.artifact["unsupported_levels"],["Medium"])
            self.assertEqual(report["student"]["levels"],list(levels))
            self.assertEqual(len(report["student"]["confusion_true_rows_predicted_columns"]),3)
            self.assertLess(report["export_max_absolute_difference"],1e-6)
            for output_index,(level,wire_class) in enumerate(zip(levels,(3,1,0))):
                artifact={**model.artifact,"w2":np.zeros((3,12)).tolist(),
                    "b2":[10. if i==output_index else 0. for i in range(3)]}
                from parameters.contracts import digest
                artifact["sha256"]=digest({k:v for k,v in artifact.items() if k!="sha256"})
                prediction=SPIClassifier(artifact).predict(np.ones(4),"fixture_features")
                self.assertEqual(prediction["level"],level)
                self.assertEqual(prediction["scheduling_class"],wire_class)
                self.assertEqual(MinimalHeader.from_bytes(MinimalHeader(1,1,1,1,
                    prediction["scheduling_class"]).to_bytes()).scheduling_class,wire_class)
            (p/"labels.json").write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError,"outside configured"):
                prepare(p/"features.npz",p/"labels.json",levels)

    def test_three_class_metrics_and_invalid_level_contracts(self):
        levels=("Low","High","Critical")
        report=metrics(np.array([[1.,0,0],[1.,0,0]]),np.array([1,2]),levels)
        self.assertEqual(report["high_critical_underpredictions"],2)
        self.assertAlmostEqual(report["rps"],.75)
        for invalid in (("Low",),("Critical","Low"),("Low","Low"),("Low","unknown")):
            with self.assertRaises(ValueError):validate_levels(invalid)


class MinimalHeaderTests(unittest.TestCase):
    def test_roundtrip_masks_identity_and_quantization(self):
        h=MinimalHeader(5,12,17,2,0,0.,0.,123456789,1000)
        self.assertEqual(len(h.to_bytes()),28)
        restored=MinimalHeader.from_bytes(h.to_bytes())
        self.assertEqual((restored.source,restored.session,restored.message,restored.scheduling_class),(5,12,17,0))
        self.assertLessEqual(abs(restored.relevance-h.relevance),1/255+1e-9)
        self.assertEqual(restored.novelty,0.)
        missing=MinimalHeader.from_bytes(MinimalHeader(1,2,3,4).to_bytes())
        self.assertIsNone(missing.novelty);self.assertIsNone(missing.scheduling_class)

    def test_query_before_commit_and_no_fake_timestamp(self):
        mission=MissionSpec("m","Look at roads.",(("road","a road"),),allowed_delay_ms=1000)
        e=VisualEvidence(embedding=np.array([1.,0.]),similarities={context_similarity_key(mission):0.},provenance={"backbone":"fixture"})
        kb=SenderMemory("ue1","m"); extractor=MinimalParameterExtractor()
        meta={"source_id":"ue1","source_numeric_id":1,"session_id":1,"message_id":1,"profile_id":1,
              "sequence":1,"capture_time_ms":123,"now_ms":200}
        r=extractor.extract_evidence(e,mission,meta,kb)
        self.assertEqual(kb.version,0);self.assertTrue(r["novelty"]["cold_start"])
        self.assertIsNone(r["capture"]["capture_time_ms"]);self.assertFalse(r["spi"]["available"])
        extractor.serialize_and_commit(e,mission,meta,kb)
        r=extractor.extract_evidence(e,mission,meta,kb)
        self.assertAlmostEqual(r["novelty"]["value"],0.)
        before=kb.version
        with self.assertRaises(ValueError):extractor.serialize_and_commit(e,mission,{**meta,"source_numeric_id":999999},kb)
        self.assertEqual(kb.version,before)


if __name__=="__main__":unittest.main()
