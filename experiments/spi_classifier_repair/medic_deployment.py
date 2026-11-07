"""Pinned image-in/four-class-out research runtime; no implicit promotion."""
from pathlib import Path
import json,time
import joblib
import numpy as np
import torch
from PIL import Image,ImageOps
from experiments.spi_classifier_repair.medic_campaign import load_encoder,encode,TokenHead,LEVELS

def probabilities(scores,temperature=1.):
    scores=np.asarray(scores,dtype=np.float64)
    if not np.isfinite(temperature) or temperature<=0 or not np.isfinite(scores).all() or scores.ndim!=2 or scores.shape[1]!=4:
        raise ValueError('finite four-class scores and positive temperature required')
    scores=scores/temperature;scores-=scores.max(-1,keepdims=True)
    values=np.exp(scores);return values/values.sum(-1,keepdims=True)

class SPIImagePredictor:
    def __init__(self,selection,identity,device='cpu',temperature=1.):
        self.selection=selection;self.device=device;self.temperature=temperature
        self.core,self.projection,self.processor,self.identity=load_encoder(selection['encoder'],device,pinned_identity=identity)
        if selection['kind']=='vector':self.head=joblib.load(selection['artifact'])
        else:
            checkpoint=torch.load(selection['artifact'],map_location='cpu',weights_only=True)
            self.head=TokenHead(checkpoint['dimension'],checkpoint['kind']).to(device)
            self.head.load_state_dict(checkpoint['state']);self.head.eval()
        self.runtime_promotion_allowed=False
    def authorize(self,receipt):
        from experiments.spi_classifier_repair.resume_store import file_hash
        checks=receipt.get('checks',{})
        required={'all_class_support','per_class_recall','export_parity','diagnostic_review','runtime_integration'}
        if (receipt.get('runtime_promotion_allowed') is not True or receipt.get('final_test_gate_passed') is not True
                or not required<=set(checks) or not all(checks[k] is True for k in required)):
            raise ValueError('SPI model has not passed release checks')
        if receipt.get('model_sha256')!=file_hash(self.selection['artifact']) or receipt.get('identity')!=self.identity:
            # Encoder loader omits only experiment label added by the cache.
            expected=dict(receipt.get('identity',{}));expected.pop('encoder_name',None)
            if receipt.get('model_sha256')!=file_hash(self.selection['artifact']) or expected!=self.identity:
                raise ValueError('SPI release artifact identity mismatch')
        if receipt.get('temperature')!=self.temperature:raise ValueError('SPI calibration identity mismatch')
        self.runtime_promotion_allowed=True
    def prepare(self,image):
        image=ImageOps.exif_transpose(image).convert('RGB');size=image.size
        picture=ImageOps.pad(image,(224,224),method=Image.Resampling.BICUBIC,color=(127,127,127))
        kwargs={'do_resize':False}
        if hasattr(self.processor,'do_center_crop'):kwargs['do_center_crop']=False
        return self.processor(images=[picture],return_tensors='pt',**kwargs)['pixel_values'].to(self.device),size
    def logits_from_features(self,global_,tokens,mask):
        if self.selection['kind']=='token':
            with torch.inference_mode():return self.head(tokens,mask).cpu().numpy()
        x=global_.cpu().numpy()
        if hasattr(self.head,'predict_proba'):
            raw=np.clip(self.head.predict_proba(x),1e-12,1);scores=np.log(raw)
        elif hasattr(self.head,'decision_function'):scores=np.asarray(self.head.decision_function(x))
        else:
            pred=self.head.predict(x);scores=np.full((len(x),4),-20.);scores[np.arange(len(x)),pred]=0
        result=np.full((len(x),4),-20.)
        for j,c in enumerate(self.head.classes_):result[:,int(c)]=scores[:,j]
        return result
    def predict_image(self,image):
        if isinstance(image,(str,Path)):
            with Image.open(image) as opened:return self.predict_image(opened)
        pixels,(width,height)=self.prepare(image)
        with torch.inference_mode():
            global_,tokens=encode(self.core,self.projection,pixels,self.selection['encoder'])
            side=int(np.sqrt(tokens.shape[1]));scale=min(224/width,224/height);w,h=width*scale,height*scale
            if side*side!=tokens.shape[1]:raise ValueError('encoder spatial grid is not square')
            yy,xx=np.meshgrid((np.arange(side)+.5)*224/side,(np.arange(side)+.5)*224/side,indexing='ij')
            mask=((xx>=(224-w)/2)&(xx<(224+w)/2)&(yy>=(224-h)/2)&(yy<(224+h)/2)).ravel()[None]
            if not mask.any():raise ValueError('unsupported aspect ratio: no valid token centers')
            logits=self.logits_from_features(global_,tokens,torch.as_tensor(mask,device=self.device))
        p=probabilities(logits,self.temperature)[0];index=int(p.argmax())
        return {'available':True,'level':LEVELS[index],'scheduling_class':3-index,'probabilities':p.tolist(),
            'logits':logits[0].tolist(),'runtime_promotion_allowed':self.runtime_promotion_allowed,
            'interpretation':'calibrated teacher agreement, not independently verified urgency'}

def fit_temperature(logits,targets):
    """Deterministic scalar calibration; training OOF only, never final test."""
    logits=np.asarray(logits);targets=np.asarray(targets)
    if logits.shape!=(len(targets),4) or not len(targets):raise ValueError('invalid calibration rows')
    values=np.exp(np.linspace(np.log(.25),np.log(4.),81));losses=[]
    for value in values:
        p=probabilities(logits,value);losses.append(float(-np.log(np.clip(p[np.arange(len(targets)),targets],1e-12,1)).mean()))
    i=int(np.argmin(losses));return {'temperature':float(values[i]),'training_oof_nll':losses[i],
        'grid':values.tolist(),'losses':losses,'scope':'training OOF selection-exposed; final test calibration reported separately'}
