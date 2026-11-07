"""Frozen validation-selected SPI student, not a VLM teacher or native urgency label."""
import hashlib
import json
from pathlib import Path
import time
from experiments.scene_array_cache.cache import file_hash, code_hash, write_json, read_image

LEVELS = ['Low', 'Medium', 'High', 'Critical']


def freeze_student(comparison, output, training_receipt):
    """Select from development metrics only; exact artifacts become an immutable bundle."""
    import shutil
    comparison, output = Path(comparison).resolve(), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    receipt = json.loads(Path(training_receipt).read_text())
    if receipt.get('cohort_size') != 4000 or receipt.get('status') != 'complete':
        raise ValueError('completed4000training receipt required; no silent historical1000 fallback')
    status = json.loads((comparison/'study_status.json').read_text())
    if len(status)!=4 or any(r['status']!='complete' for r in status):
        raise ValueError('four balance studies must have completed')
    pairs = json.loads((comparison/'frozen_pair_selections.json').read_text())
    if len(pairs)!=48: raise ValueError('complete48pair development selection required')
    winner = sorted(pairs, key=lambda r:(-r['development']['macro_f1'],r['encoder'],r['head'],r['balance']))[0]
    relative = Path(winner['balance'])/winner['encoder']/(winner['head']+('.pt' if winner['kind']=='token' else '.joblib'))
    artifact = comparison/relative
    if not artifact.is_file(): raise FileNotFoundError('selected student artifact missing: '+str(artifact))
    identity = comparison/'features'/winner['encoder']/'identity.json'
    if not identity.is_file(): raise FileNotFoundError('exact encoder identity missing')
    head = output/('head'+artifact.suffix)
    shutil.copy2(artifact, head); shutil.copy2(identity, output/'encoder.json')
    # Current study exports unscaled sklearn vectors and fixed TokenHead definitions.
    bundle = {'schema':'image_pool_spi_v1','levels':LEVELS,'encoder':winner['encoder'],
        'kind':winner['kind'],'head':winner['head'],'balance':winner['balance'],
        'selection_basis':'development macroF1 only; no test-based choice',
        'development':winner['development'],'head_file':head.name,'head_sha256':file_hash(head),
        'encoder_file':'encoder.json','encoder_identity_sha256':file_hash(output/'encoder.json'),
        'training_receipt':receipt,'training_receipt_sha256':file_hash(training_receipt),
        'classifier_source_sha256':code_hash(Path(__file__).parents[1]/'spi_classifier_repair/medic_campaign.py'),
        'scope':'experimental SPI predictions, not independent urgency truth'}
    write_json(output/'bundle.json',bundle)
    return output/'bundle.json'


class Student:
    def __init__(self, bundle_path, device):
        import joblib
        import torch
        from experiments.spi_classifier_repair.medic_campaign import load_encoder, TokenHead
        self.bundle_path = Path(bundle_path); self.bundle = json.loads(self.bundle_path.read_text())
        self.device=device
        if self.bundle.get('schema')!='image_pool_spi_v1' or self.bundle.get('levels')!=LEVELS:
            raise ValueError('unsupported SPI bundle')
        head_path=self.bundle_path.parent/self.bundle['head_file']
        identity_path=self.bundle_path.parent/self.bundle['encoder_file']
        if file_hash(head_path)!=self.bundle['head_sha256'] or file_hash(identity_path)!=self.bundle['encoder_identity_sha256']:
            raise ValueError('SPI bundle artifact checksum mismatch')
        source=Path(__file__).parents[1]/'spi_classifier_repair/medic_campaign.py'
        if file_hash(source)!=self.bundle['classifier_source_sha256']:
            # Text line endings are allowed to vary across runtime snapshots.
            normalized=hashlib.sha256(source.read_text(encoding='utf8').encode()).hexdigest()
            if normalized!=self.bundle['classifier_source_sha256']:
                raise ValueError('student architecture source mismatch')
        identity=json.loads(identity_path.read_text())
        start=time.perf_counter()
        self.core,self.projection,self.processor,self.identity=load_encoder(self.bundle['encoder'],device,pinned_identity=identity)
        self.encoder_load_seconds=time.perf_counter()-start
        start=time.perf_counter()
        self.token=self.bundle['kind']=='token'
        if self.token:
            state=torch.load(head_path,map_location='cpu',weights_only=False)
            self.head=TokenHead(state['dimension'],state['kind'])
            self.head.load_state_dict(state['state']);self.head=self.head.to(device).eval()
        else:
            self.head=joblib.load(head_path)
            if list(self.head.classes_)!=list(range(4)): raise ValueError('student must retain all4classes')
        self.head_load_seconds=time.perf_counter()-start
        self.identity_hash=file_hash(self.bundle_path)

    def sync(self):
        if self.device.startswith('cuda'):
            import torch
            torch.cuda.synchronize(self.device)

    def predict(self,path):
        import numpy as np
        import torch
        from PIL import ImageOps,Image
        from experiments.spi_classifier_repair.medic_campaign import encode
        self.sync();start=time.perf_counter();image=read_image(path)
        width,height=image.size
        padded=ImageOps.pad(image,(224,224),method=Image.Resampling.BICUBIC,color=(127,127,127))
        kw={'do_resize':False}
        if hasattr(self.processor,'do_center_crop'):kw['do_center_crop']=False
        pixels=self.processor(images=[padded],return_tensors='pt',**kw)['pixel_values'].to(self.device)
        self.sync();pre=time.perf_counter()
        with torch.inference_mode():
            vector,tokens=encode(self.core,self.projection,pixels,self.bundle['encoder'])
        self.sync();encoded=time.perf_counter()
        if self.token:
            side=int(np.sqrt(tokens.shape[1]))
            if side*side!=tokens.shape[1]: raise ValueError('nonspatial token grid')
            yy,xx=np.meshgrid((np.arange(side)+.5)*224/side,(np.arange(side)+.5)*224/side,indexing='ij')
            scale=min(224/width,224/height);w,h=width*scale,height*scale
            mask=torch.tensor(((xx>=(224-w)/2)&(xx<(224+w)/2)&(yy>=(224-h)/2)&(yy<(224+h)/2)).reshape(1,-1),device=self.device)
            if not bool(mask.any()):raise ValueError('unsupported extreme aspect ratio')
            with torch.inference_mode():scores=self.head(tokens,mask).softmax(-1).cpu().numpy()[0]
            predicted=int(scores.argmax());kind='uncalibrated_softmax'
        else:
            x=vector.cpu().numpy();predicted=int(self.head.predict(x)[0])
            if hasattr(self.head,'predict_proba'):
                scores=self.head.predict_proba(x)[0];kind='uncalibrated_predict_proba'
            elif hasattr(self.head,'decision_function'):
                scores=np.asarray(self.head.decision_function(x)[0]);kind='decision_scores_not_probabilities'
            else:scores=None;kind='class_only'
        self.sync();end=time.perf_counter()
        if predicted not in range(4) or scores is not None and (scores.shape!=(4,) or not np.isfinite(scores).all()):
            raise ValueError('invalid SPI prediction')
        return {'level':LEVELS[predicted],'class_id':predicted,'available':True,
            'scores':None if scores is None else scores.tolist(),'score_type':kind,
            'student_bundle_sha256':self.identity_hash,'reference_type':'student_prediction',
            'timing_ms':{'load_preprocess':(pre-start)*1000,'encoder':(encoded-pre)*1000,
                         'head_transfer':(end-encoded)*1000,'total':(end-start)*1000},
            'model_domain_warning':'disaster-trained SPI; cross-domain urgency validity unestablished'}
