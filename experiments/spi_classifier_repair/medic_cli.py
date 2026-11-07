"""Single/batch image parameter CLI for a verified SPI bundle."""
import argparse,hashlib,json,time
from pathlib import Path
from PIL import Image
from parameters.contracts import MissionSpec
from parameters.extractor import SenderMemory
from parameters.minimal_parameters import FinalParameterExtractor
from parameters.minimal_vision import MinimalVisionProvider
from parameters.scene_measurements import YOLOWorldSceneProvider
from parameters.parameter_cli import load_memory,save_memory
from parameters.artifacts import write_json
from experiments.spi_classifier_repair.medic_deployment import SPIImagePredictor

def load_bundle(path,device='cpu',research=False):
    path=Path(path).resolve();manifest=json.loads((path/'bundle.json').read_text())
    for name,expected in manifest['files'].items():
        file=(path/name).resolve()
        if not file.is_relative_to(path):raise ValueError('invalid bundle path')
        actual=hashlib.sha256(file.read_bytes()).hexdigest()
        if actual!=expected:raise ValueError('SPI bundle checksum mismatch: '+name)
    selection=json.loads((path/'selection.json').read_text());selection['artifact']=str(path/selection['artifact'])
    receipt=json.loads((path/'readiness.json').read_text())
    if not research and receipt.get('runtime_promotion_allowed') is not True:raise ValueError('bundle failed readiness; no production extraction allowed')
    predictor=SPIImagePredictor(selection,json.loads((path/'encoder_identity.json').read_text()),device,receipt['temperature'])
    if not research:predictor.authorize(receipt)
    return predictor

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',required=True);p.add_argument('--images',nargs='+',required=True)
    p.add_argument('--mission',required=True);p.add_argument('--clip-checkpoint',required=True);p.add_argument('--yolo-checkpoint',required=True)
    p.add_argument('--device',choices=['cpu','cuda:0'],default='cpu');p.add_argument('--ue-id',default='ue01');p.add_argument('--kb')
    p.add_argument('--metadata');p.add_argument('--allowed-delay-ms',type=int);p.add_argument('--output',required=True)
    p.add_argument('--research',action='store_true',help='preserve audit scores; unavailable SPI bit remains clear')
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError('choose a new output path')
    mission=MissionSpec('operator_mission',a.mission,(('mission',a.mission),),allowed_delay_ms=a.allowed_delay_ms)
    memory=load_memory(a.kb,a.ue_id,mission.mission_id,mission.version,capacity=128)
    metadata=json.loads(Path(a.metadata).read_text()) if a.metadata else {}
    predictor=load_bundle(a.bundle,a.device,a.research);visual=MinimalVisionProvider(a.clip_checkpoint,[mission],a.device)
    detector=YOLOWorldSceneProvider(a.yolo_checkpoint,device=a.device)
    extractor=FinalParameterExtractor(visual,scene_provider=detector,spi_image_provider=predictor);results=[]
    try:
        for image_path in a.images:
            sha=hashlib.sha256(Path(image_path).read_bytes()).hexdigest();capture=metadata.get(sha,{})
            sequence=memory.last_sequence+1
            meta={'source_id':a.ue_id,'source_numeric_id':1,'session_id':1,'message_id':sequence+1,'profile_id':1,
                'sequence':sequence,'capture_time_ms':capture.get('capture_time_ms'),'timestamp_trusted':capture.get('timestamp_trusted') is True}
            with Image.open(image_path) as image:bundle,payload=extractor.transmit_batch([image],[mission],[meta],[memory])[0]
            results.append({'image_sha256':sha,'parameters':bundle['compact'],'header_hex':payload.hex(),'header_bytes':len(payload),
                            'spi_audit':bundle.get('spi_audit'),'research_only':a.research})
            save_memory(a.kb,memory,mission.version)
        write_json(out,results)
    finally:visual.close()

if __name__=='__main__':main()
