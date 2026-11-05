import json,tempfile,unittest
from pathlib import Path
import numpy as np
from experiments.spi_classifier_repair.resume_store import ResumeStore,atomic_json
from experiments.spi_classifier_repair.medic_deployment import fit_temperature,probabilities
from parameters.contracts import MissionSpec
from parameters.extractor import SenderMemory,VisualEvidence
from parameters.minimal_parameters import FinalParameterExtractor,ExtendedParameterHeader
from parameters.pilot_parameters import context_similarity_key

class ResumeTests(unittest.TestCase):
    def test_complete_only_and_hash_validation(self):
        with tempfile.TemporaryDirectory() as root:
            store=ResumeStore(root);spec={'images':[('a','b')],'version':1}
            self.assertIsNone(store.read('stage',spec));directory=store.directory('stage',spec)
            atomic_json(directory/'value.json',{'done':True});self.assertIsNone(store.read('stage',spec))
            store.complete('stage',spec,[directory/'value.json'],{'value':4})
            self.assertEqual(store.read('stage',spec)['value']['value'],4)
            self.assertIsNone(store.read('stage',{**spec,'version':2}))
            (directory/'value.json').write_text('corrupted')
            with self.assertRaises(ValueError):store.read('stage',spec)
    def test_invalid_artifact_scope(self):
        with tempfile.TemporaryDirectory() as root:
            store=ResumeStore(root);file=Path(root)/'outside';file.write_text('x')
            with self.assertRaises(ValueError):store.complete('stage',{'x':1},[file],{})
    def test_temperature_is_valid(self):
        logits=np.eye(4)*2;y=np.arange(4);cal=fit_temperature(logits,y)
        self.assertTrue(.25<=cal['temperature']<=4)
        self.assertTrue(np.allclose(probabilities(logits,cal['temperature']).sum(1),1))

class ImageRuntimeTests(unittest.TestCase):
    def test_batch_wire_gate_and_causal_commit(self):
        mission=MissionSpec('m','Inspect visible problems.',(('problem','visible problem'),))
        class Visual:
            def extract_batch(self,images,missions):
                return [VisualEvidence(embedding=np.array([1.,0.]),similarities={context_similarity_key(m):.5},provenance={'backbone':'fixture'}) for m in missions]
        class Predictor:
            ready=False
            def predict_image(self,image):return {'available':True,'level':'High','scheduling_class':1,'runtime_promotion_allowed':self.ready}
        model=Predictor();extractor=FinalParameterExtractor(Visual(),spi_image_provider=model)
        memory=SenderMemory('ue','m',capacity=128)
        meta={'source_id':'ue','source_numeric_id':1,'session_id':1,'message_id':1,'profile_id':1,'sequence':1}
        result=extractor.transmit_batch([object()],[mission],[meta],[memory])[0]
        self.assertIsNone(ExtendedParameterHeader.from_bytes(result[1]).core.scheduling_class)
        self.assertEqual(memory.version,1)
        model.ready=True;meta={**meta,'sequence':2,'message_id':2}
        result=extractor.transmit_batch([object()],[mission],[meta],[memory])[0]
        self.assertEqual(ExtendedParameterHeader.from_bytes(result[1]).core.scheduling_class,1)
        self.assertEqual(memory.version,2);self.assertEqual(result[0]['novelty']['value'],0.)
    def test_mutually_exclusive_spi_providers(self):
        with self.assertRaises(ValueError):FinalParameterExtractor(spi_classifier=object(),spi_image_provider=object())

if __name__=='__main__':unittest.main()
