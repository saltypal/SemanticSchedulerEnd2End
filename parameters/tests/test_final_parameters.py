"""Numerical scene, wire and causal-memory checks; no learned accuracy claims."""
import math
import unittest
import numpy as np
from parameters.contracts import MissionSpec
from parameters.extractor import SenderMemory, VisualEvidence
from parameters.minimal_parameters import MinimalHeader, ExtendedParameterHeader, MinimalParameterExtractor, FinalParameterExtractor
from parameters.pilot_parameters import context_similarity_key
from parameters.scene_measurements import SceneMeasurements, measure_detections, YOLOWorldSceneProvider
from parameters.spi_final_campaign import CONTROLS, execute as execute_final_campaign
from parameters.spi_labels import load_rubric, label_prompt
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import hashlib
import zipfile
from parameters.spi import SPIClassifier
from parameters.spi_model_selection import select, support_gate
from parameters.parameter_vlm_review import corrupt, parse_review
from parameters.parameter_cli import load_memory, save_memory
from parameters.minimal_vision import MinimalVisionProvider


class SceneTests(unittest.TestCase):
    def test_exact_union_clipping_alias_and_rejected(self):
        rows=[{'label':'people','box':[0,0,5,5],'score':.9},
              {'label':'cars','box':[3,0,8,5],'score':.8},
              {'label':'car','box':[-2,-2,1,1],'score':.5},
              {'label':'person','box':[0,0,math.nan,3],'score':.9},
              {'label':'person','box':[2,2,2,4],'score':.9}]
        s=measure_detections(rows,10,10)
        self.assertEqual((s.instances,s.categories),(3,2))
        self.assertAlmostEqual(s.largest_box_fraction,.25)
        self.assertAlmostEqual(s.union_box_coverage,.4)
        self.assertEqual(s.audit['rejected_boxes'],2)
        self.assertEqual(s.audit['category_counts'],{'car':2,'person':1})

    def test_empty_and_missing_are_different(self):
        empty=measure_detections([],10,10)
        missing=SceneMeasurements.unavailable('detector failed')
        self.assertEqual(empty.instances,0);self.assertEqual(empty.categories,0)
        self.assertEqual(empty.union_box_coverage,0);self.assertIsNone(empty.largest_box_fraction)
        self.assertIsNone(missing.instances);self.assertIsNone(missing.union_box_coverage)
        with self.assertRaises(ValueError):SceneMeasurements(0,1,None,0)
        with self.assertRaises(ValueError):SceneMeasurements(0,0,.2,0)
        with self.assertRaises(ValueError):SceneMeasurements(1,0,.2,.2)
        with self.assertRaises(ValueError):SceneMeasurements(1,1,.8,.2)

    def test_saturation_and_nested_overlap(self):
        rows=[{'label':'person','box':[0,0,10,10]}, {'label':'person','box':[2,2,5,5]}]
        s=measure_detections(rows,10,10,max_detections=2)
        self.assertTrue(s.saturated);self.assertEqual(s.union_box_coverage,1.)
        core=MinimalHeader(1,1,1,1)
        self.assertTrue(ExtendedParameterHeader.from_bytes(ExtendedParameterHeader(core,s).to_bytes()).scene.saturated)

    def test_unknown_class_is_rejected_when_vocabulary_is_known(self):
        rows=[{'label':'person','box':[0,0,3,3]}, {'label':'imaginary','box':[0,0,5,5]}]
        scene=measure_detections(rows,10,10,provenance={'vocabulary':['person']})
        self.assertEqual(scene.instances,1);self.assertEqual(scene.audit['rejected_boxes'],1)

    def test_wire_roundtrip_masks_quantization_and_v1(self):
        core=MinimalHeader(1,2,3,4,2,.2,.7,100,200,False)
        scene=SceneMeasurements(2,1,.3,.45,False,'ok')
        raw=ExtendedParameterHeader(core,scene).to_bytes()
        self.assertEqual(len(raw),35);self.assertEqual(raw[0],2)
        back=ExtendedParameterHeader.from_bytes(raw)
        self.assertEqual(back.core.scheduling_class,2)
        self.assertEqual((back.scene.instances,back.scene.categories),(2,1))
        self.assertLessEqual(abs(back.scene.union_box_coverage-.45),.5/255+1e-8)
        self.assertEqual(len(core.to_bytes()),28)
        self.assertEqual(MinimalHeader.from_bytes(core.to_bytes()).scheduling_class,2)
        miss=ExtendedParameterHeader(core,SceneMeasurements.unavailable('x'))
        self.assertIsNone(ExtendedParameterHeader.from_bytes(miss.to_bytes()).scene.instances)
        bad=bytearray(raw);bad[28]|=0x80
        with self.assertRaises(ValueError):ExtendedParameterHeader.from_bytes(bad)

    def test_batch_is_read_only_and_transmission_is_serial(self):
        mission=MissionSpec('m','Inspect roads.',(('road','road'),),allowed_delay_ms=1000)
        class Provider:
            def extract_batch(self,images,missions):
                return [VisualEvidence(embedding=np.array([1.,0.]),similarities={context_similarity_key(m):.3},
                    provenance={'backbone':'fixture'}) for m in missions]
        class Detector:
            def measure(self,image):return measure_detections([],10,10)
        memory=SenderMemory('ue','m',capacity=32)
        extractor=MinimalParameterExtractor(Provider(),scene_provider=Detector())
        meta={'source_id':'ue','source_numeric_id':1,'session_id':1,'message_id':1,'profile_id':1,
              'sequence':1,'capture_time_ms':100,'timestamp_trusted':True,'now_ms':200}
        second={**meta,'sequence':2,'message_id':2}
        before=extractor.extract_batch([1,2],[mission,mission],[meta,second],[memory,memory])
        self.assertEqual(memory.version,0)
        self.assertTrue(all(row['novelty']['cold_start'] for row in before))
        out=extractor.transmit_batch([1,2],[mission,mission],[meta,second],[memory,memory])
        self.assertEqual(memory.version,2);self.assertEqual(len(out[0][1]),35)
        self.assertTrue(out[0][0]['novelty']['cold_start'])
        self.assertTrue(out[0][0]['compact']['novelty_cold_start'])
        self.assertFalse(out[1][0]['compact']['novelty_cold_start'])
        self.assertAlmostEqual(out[1][0]['novelty']['value'],0.)
        self.assertIsNone(out[0][0]['compact']['spi'])

    def test_v2_prompt_has_balanced_definitions_and_parser_contract(self):
        rubric=load_rubric(Path(__file__).resolve().parents[1]/'config/spi_rubric_v2.json')
        prompt=label_prompt(rubric)
        self.assertEqual({name:sum(x[0]==name for x in CONTROLS) for name in rubric['levels']},
                         {'Low':3,'Medium':3,'High':3,'Critical':3})
        for name in rubric['levels']:self.assertIn(name,prompt)
        self.assertNotIn('Example: {"level":"Low"',prompt)

    def test_four_class_selection_export_and_sealed_labels(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);rows=[];vectors=[];rng=np.random.default_rng(5)
            for split,count in [('train',12),('validation',5),('test',1)]:
                for level,name in enumerate(('Low','Medium','High','Critical')):
                    for j in range(count):
                        identity=f'{split}-{name}-{j}'
                        rows.append({'image_id':identity,'image_sha256':identity,'group':identity,'split':split,
                            'available':True,'level':name,'label_source':'vlm_pseudo','label_unit':'whole_image',
                            'rubric_sha256':'fixture_v2'})
                        vectors.append(np.eye(4)[level]+rng.normal(0,.02,4))
            np.savez_compressed(root/'features.npz',image_ids=np.array([r['image_id'] for r in rows]),
                image_sha256=np.array([r['image_sha256'] for r in rows]),embeddings=np.asarray(vectors,np.float32),
                feature_signature=np.array('fixture'))
            for row in rows:
                if row['split']=='test':row.update(level='POISONED',available='POISONED')
            (root/'labels.json').write_text(json.dumps(rows))
            self.assertTrue(support_gate(rows)['passed'])
            report=select(root/'features.npz',root/'labels.json',root/'result')
            self.assertFalse(report['test_evaluated'])
            self.assertLess(report['export_max_absolute_difference'],1e-5)
            model=SPIClassifier.load(root/'result/models/spi.json')
            self.assertEqual(model.predict(np.array([1.,0,0,0]),'fixture')['level'],'Low')

    def test_review_parser_and_corrupt_controls(self):
        good={'assessable':True,'consistent':True,'unsupported_fields':[],'evidence':'ok','reason':None}
        self.assertTrue(parse_review(json.dumps(good))[0]['consistent'])
        with self.assertRaises(ValueError):parse_review(json.dumps({**good,'consistent':None}))
        candidate={'relevance':.2,'novelty':.3,'scene':{'instances':2,'categories':1,
            'largest_box_fraction':.2,'union_box_coverage':.3}}
        for i in range(8):self.assertNotEqual(corrupt(candidate,i),candidate)

    def test_persisted_kb_rejects_different_ue_and_mission(self):
        with TemporaryDirectory() as directory:
            path=Path(directory)/'kb.json';memory=SenderMemory('ue','m',capacity=32)
            evidence=VisualEvidence(embedding=np.ones(512),provenance={'backbone':'fixture'})
            memory.commit(evidence,3);save_memory(path,memory,'version-a')
            loaded=load_memory(path,'ue','m','version-a')
            self.assertEqual((loaded.version,loaded.last_sequence,len(loaded.entries)),(1,3,1))
            with self.assertRaises(ValueError):load_memory(path,'other','m','version-a')
            with self.assertRaises(ValueError):load_memory(path,'ue','m','version-b')

    def test_final_extractor_emits_v2_even_without_detector(self):
        m=MissionSpec('m','Inspect roads.',(('road','road'),))
        e=VisualEvidence(embedding=np.array([1.,0.]),provenance={'backbone':'fixture'})
        meta={'source_id':'ue','source_numeric_id':1,'session_id':1,'message_id':1,'profile_id':1}
        result=FinalParameterExtractor().extract_evidence(e,m,meta,SenderMemory('ue','m',32))
        self.assertEqual(result['schema'],'better_minimal_parameters_v2')
        self.assertIsNone(result['compact']['scene']['instances'])
        self.assertEqual(len(result['header'].to_bytes()),35)

    def test_invalid_context_and_future_time_preserve_other_fields(self):
        m=MissionSpec('m','Inspect roads.',(('road','road'),))
        e=VisualEvidence(embedding=np.array([1.,0.]),similarities={context_similarity_key(m):'bad'},
                         provenance={'backbone':'fixture','context_reason':'invalid text embedding'})
        meta={'source_id':'ue','source_numeric_id':1,'session_id':1,'message_id':1,'profile_id':1,
              'capture_time_ms':200,'timestamp_trusted':True,'now_ms':100}
        result=FinalParameterExtractor().extract_evidence(e,m,meta,SenderMemory('ue','m',32))
        self.assertFalse(result['relevance']['available']);self.assertEqual(result['relevance']['reason'],'invalid text embedding')
        self.assertEqual(result['novelty']['value'],1.);self.assertTrue(result['novelty']['cold_start'])
        self.assertIsNone(result['capture']['capture_time_ms'])
        self.assertIn('later',result['capture']['capture_reason'])

    def test_text_preflight_does_not_skip_visual_preflight(self):
        class Reviewer:
            provenance={'model':'test fixture'}
            _finite_preflight_passed=False
            def __init__(self):self.text=0;self.visual=0;self.closed=False
            def review_structured(self,images,payload,prompt,parser,case_id,raw_dir,*,protocol,text_only_diagnostic=False,text_only_prefix=None):
                if text_only_diagnostic:
                    assert 'candidate and reference' not in text_only_prefix
                    assert 'Use image and fixed rubric only' not in prompt
                    self.text+=1;self._finite_preflight_passed=True
                    level=CONTROLS[self.text-1][0]
                else:
                    self.assertion=(self._finite_preflight_passed is False and set(payload)=={'instruction'})
                    self._finite_preflight_passed=True;self.visual+=1;level='Low'
                parsed,_=parser(json.dumps({'level':level,'evidence':'fixture','reason':None}))
                return {'status':'complete','parsed':parsed}
            def close(self):self.closed=True
        with TemporaryDirectory() as directory:
            root=Path(directory);data=b'test image bytes';sha=hashlib.sha256(data).hexdigest()
            rows=[{'image_id':'a','image_sha256':sha,'image_path':'images/a.png',
                   'group':'g','split':'train','source':'fixture'}]
            (root/'panel.json').write_text(json.dumps(rows));(root/'features.npz').write_bytes(b'fixture')
            with zipfile.ZipFile(root/'images.zip','w') as z:z.writestr('images/a.png',data)
            (root/'input_manifest.json').write_text(json.dumps({'files':{name:hashlib.sha256((root/name).read_bytes()).hexdigest()
                for name in ('panel.json','features.npz','images.zip')}}))
            reviewer=Reviewer();execute_final_campaign(root,root/'result',reviewer)
            self.assertTrue(reviewer.assertion);self.assertEqual((reviewer.text,reviewer.visual),(12,1))
            self.assertTrue(reviewer.closed)
            self.assertEqual(json.loads((root/'result/status.json').read_text())['status'],'labels_complete')

    def test_zero_context_embedding_is_unavailable_not_nan(self):
        import torch
        class Tokenizer:
            bos_token_id=1;eos_token_id=2;pad_token_id=0
            def __call__(self,text,add_special_tokens=False):return {'input_ids':[3]}
        class Model:
            config=type('Config',(),{'text_config':type('Text',(),{'max_position_embeddings':77})()})()
            def get_text_features(self,**kwargs):return torch.zeros((1,512))
        class Fake:
            device='cpu';processor=type('Processor',(),{'tokenizer':Tokenizer()})();model=Model()
            def load(self):pass
        fake=Fake();fake.torch=torch
        provider=MinimalVisionProvider.__new__(MinimalVisionProvider);provider.provider=fake;provider.context_cache={}
        m=MissionSpec('m','Inspect roads.',(('road','road'),))
        with self.assertRaisesRegex(ValueError,'zero or nonfinite'):provider.context_embedding(m)

    def test_missing_yolo_checkpoint_is_explicitly_unavailable(self):
        from PIL import Image
        with TemporaryDirectory() as directory:
            producer=YOLOWorldSceneProvider(Path(directory)/'missing.pt')
            result=producer.measure(Image.new('RGB',(8,8)))
            self.assertEqual(result.status,'unavailable')
            self.assertIsNone(result.instances)
            self.assertIn('checkpoint not found',result.reason)

    def test_legacy_four_class_artifact_still_loads(self):
        from parameters.contracts import digest
        artifact={'schema':'better_spi_head_v1','levels':['Low','Medium','High','Critical'],
            'input_dimension':4,'hidden_dimension':4,'feature_signature':'fixture','rubric_sha256':'fixture',
            'w1':np.eye(4).tolist(),'b1':[0.,0.,0.,0.],
            'w2':np.eye(4).tolist(),'b2':[0.,0.,0.,0.]}
        artifact['sha256']=digest(artifact)
        prediction=SPIClassifier(artifact).predict(np.array([1.,0,0,0]),'fixture')
        self.assertEqual((prediction['level'],prediction['scheduling_class']),('Low',3))

if __name__=='__main__':unittest.main()
