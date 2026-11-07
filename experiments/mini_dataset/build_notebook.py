"""Build Mini dataset enrichment from the shared image-pool notebook."""
import json
from pathlib import Path
import nbformat

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent

def build(model_dir):
    cloud_train=model_dir=='cloud'
    model_dir=Path(model_dir)
    notebook=nbformat.read(ROOT/'experiments/image_pool/Image_Pool_Caching.ipynb',as_version=4)
    notebook.cells[0].source='# Mini dataset — provisional SPI enrichment of 6,000 images\n\nCLIP + balanced logistic regression, selected on development data only. VLM pseudo-labels are not independent urgency truth. Training-exposed images remain development data. Reuse the verified scene cache; do not rerun detection.'
    setup=next(c for c in notebook.cells if c.cell_type=='code')
    setup.source=setup.source.replace("SCENE_ONLY=os.environ.get('IMAGE_POOL_SCENE_ONLY','0')=='1'","SCENE_ONLY=False")
    # CLIP image inference uses Transformers; the unrelated OpenAI Git package failed previously.
    setup.source=setup.source.replace("'git+https://github.com/openai/CLIP.git@a1d071733d7111c9c014f024669f959182114e33'", "'ftfy'")
    head={} if cloud_train else json.loads((model_dir/'head.json').read_text())
    encoder={} if cloud_train else json.loads((model_dir/'encoder.json').read_text())
    receipt={} if cloud_train else json.loads((model_dir/'training_receipt.json').read_text())
    for cell in notebook.cells:
        if cell.cell_type!='code':continue
        if 'with stage(RUN,\'02_model_selection\')' in cell.source:
            prefix=cell.source.split("with stage(RUN,'02_model_selection'):")[0]
            cell.source=prefix+f'''\nimport joblib,numpy as np
from sklearn.linear_model import LogisticRegression
with stage(RUN,'02_provisional_model'):
    write_json(RUN/'config.json',CONFIG)
    PORTABLE={head!r}
    model=LogisticRegression(C=PORTABLE['C'],class_weight='balanced',max_iter=2000)
    model.coef_=np.array(PORTABLE['coef']);model.intercept_=np.array(PORTABLE['intercept'])
    model.classes_=np.array(PORTABLE['classes']);model.n_iter_=np.array(PORTABLE['n_iter'])
    model.n_features_in_=model.coef_.shape[1]
    directory=RUN/'student';directory.mkdir(exist_ok=True)
    joblib.dump(model,directory/'head.joblib')
    write_json(directory/'encoder.json',{encoder!r})
    write_json(directory/'training_receipt.json',{receipt!r})
    write_json(directory/'bundle.json',{{'schema':'image_pool_spi_v1','levels':['Low','Medium','High','Critical'],
      'encoder':'clip_b32','kind':'vector','head':'logistic','head_file':'head.joblib',
      'head_sha256':file_hash(directory/'head.joblib'),'encoder_file':'encoder.json',
      'encoder_identity_sha256':file_hash(directory/'encoder.json'),
      'classifier_source_sha256':code_hash(BOOT/'experiments/spi_classifier_repair/medic_campaign.py'),
      'scope':'USER APPROVED PROVISIONAL classifier; full study incomplete; pseudo-label supervision'}})
    BUNDLE=directory/'bundle.json'
'''
            if cloud_train:cell.source=prefix+"\nBUNDLE=None\nwrite_json(RUN/'config.json',CONFIG)\nprint('Provisional classifier will be trained after acquisition',flush=True)\n"
        if "with stage(RUN,'04_scene_cache'):" in cell.source:
            cell.source='''import shutil
with stage(RUN,'04_existing_scene_cache'):
    matches=[]
    for path in Path('/kaggle/input').rglob('pool_manifest.json'):
        try:
            cohort=json.loads(path.read_text())
            if len(cohort)==6000 and (path.parent/'scene/scene_records.jsonl').is_file():matches.append(path.parent)
        except (ValueError,OSError):continue
    if len(matches)!=1:raise RuntimeError('Mount exactly one completed 6000-image scene run; found '+str(matches))
    SCENE_ROOT=matches[0]
    shutil.copytree(SCENE_ROOT/'scene',RUN/'scene',dirs_exist_ok=True)
    WEIGHTS=None
    print('REUSED 6000-image scene cache',SCENE_ROOT,flush=True)
'''
        if "profile(ROWS,WEIGHTS,RUN/'scene',('cpu','cuda:0'))" in cell.source:
            cell.source=cell.source.replace("profile(ROWS,WEIGHTS,RUN/'scene',('cpu','cuda:0'))", "print('Scene inference not repeated; prior cost records retained',flush=True)")
    if cloud_train:
        index=next(i for i,c in enumerate(notebook.cells) if c.cell_type=='code' and "with stage(RUN,'05_spi_cache')" in c.source)
        train_source=(HERE/'train.py').read_text()
        training='''import importlib.util,gc,sqlite3,io
import numpy as np,torch
from PIL import Image,ImageOps
from experiments.spi_classifier_repair.medic_campaign import load_encoder,encode
from experiments.scene_array_cache.cache import read_image
with stage(RUN,'mini_training'):
    candidates=list(Path('/kaggle/input').rglob('mini_training_records.json'))
    if len(candidates)!=1:raise RuntimeError('Mount exactly one Mini training-label input')
    truth=json.loads(candidates[0].read_text());by_id={r['image_id']:r for r in truth}
    fitting=[];vectors=[];identifiers=[]
    core,projection,processor,identity=load_encoder('clip_b32','cuda:0')
    directory=RUN/'student';directory.mkdir(exist_ok=True)
    write_json(directory/'encoder.json',identity)
    cache=RUN/'global_embeddings';cache.mkdir(exist_ok=True)
    write_json(cache/'encoder.json',identity)
    db=sqlite3.connect(cache/'vectors.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS vectors(hash TEXT PRIMARY KEY,value BLOB)')
    for number,row in enumerate(ROWS):
        if row['dataset']!='crisismmd' or row['source_id'] not in by_id:continue
        reference=by_id[row['source_id']]
        if reference['image_sha256']!=row['image_sha256']:raise ValueError('training image hash mismatch')
        key=row['image_sha256'];saved=db.execute('SELECT value FROM vectors WHERE hash=?',(key,)).fetchone()
        if saved:vector=np.load(io.BytesIO(saved[0]),allow_pickle=False)
        else:
            image=ImageOps.pad(read_image(row['image_path']),(224,224),method=Image.Resampling.BICUBIC,color=(127,127,127))
            kw={'do_resize':False}
            if hasattr(processor,'do_center_crop'):kw['do_center_crop']=False
            pixels=processor(images=[image],return_tensors='pt',**kw)['pixel_values'].to('cuda:0')
            with torch.inference_mode():vector=encode(core,projection,pixels,'clip_b32')[0].cpu().numpy()[0]
            stream=io.BytesIO();np.save(stream,vector)
            with db:db.execute('INSERT OR REPLACE INTO vectors VALUES(?,?)',(key,stream.getvalue()))
        fitting.append(reference);vectors.append(vector);identifiers.append(reference['image_id'])
        if len(fitting)%100==0:print('MINI TRAINING FEATURES',len(fitting),'/',len(truth),flush=True)
    if len(fitting)!=len(truth):raise ValueError('training cohort not fully acquired')
    db.close()
    features=directory/'training_vectors.npz'
    np.savez_compressed(features,embeddings=np.array(vectors),image_ids=np.array(identifiers))
    write_json(directory/'training_records.json',fitting)
    spec=importlib.util.spec_from_file_location('mini_train',RUN/'mini_train.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    receipt=module.train(features,directory/'training_records.json',directory)
    write_json(directory/'bundle.json',{'schema':'image_pool_spi_v1','levels':['Low','Medium','High','Critical'],
      'encoder':'clip_b32','kind':'vector','head':'logistic','head_file':'head.joblib',
      'head_sha256':file_hash(directory/'head.joblib'),'encoder_file':'encoder.json',
      'encoder_identity_sha256':file_hash(directory/'encoder.json'),
      'classifier_source_sha256':code_hash(BOOT/'experiments/spi_classifier_repair/medic_campaign.py'),
      'scope':'USER APPROVED PROVISIONAL classifier; full study incomplete; pseudo-label supervision'})
    BUNDLE=directory/'bundle.json'
    del core,projection,processor;gc.collect();torch.cuda.empty_cache()
    print('MINI MODEL TRAINED',receipt['development_macro_f1'],flush=True)
'''
        notebook.cells.insert(index,nbformat.v4.new_code_cell("(RUN/'mini_train.py').write_text("+repr(train_source)+",encoding='utf8')\n"+training))
    notebook.cells.append(nbformat.v4.new_code_cell('''import tarfile,hashlib
with stage(RUN,'09_mini_package'):
    if not RESULT['all_complete'] or len(ROWS)!=6000:raise RuntimeError('Mini dataset requires 6000 complete records')
    write_json(RUN/'portable_image_index.json',[{'image_id':r['image_id'],'image_sha256':r['image_sha256'],
        'relative_path':str(Path(r['image_path']).relative_to(RUN))} for r in ROWS])
    (RUN/'README_MINI.txt').write_text('Mini dataset: provisional SPI predictions and detector scene evidence. Resolve images relative to the extracted Mini_dataset folder using portable_image_index.json. Training-exposed CrisisMMD groups are development data; do not treat the full pool as an independent final test. Keep evaluator labels, IDs, datasets and split metadata outside scheduler model inputs. Relevance and causal novelty are simulator-generated. Scores are uncalibrated. See student/training_receipt.json for held-out classifier metrics.')
    target=RUN.parent/'Mini_dataset_6000.tar'
    with tarfile.open(target,'w') as archive:
        archive.add(RUN,arcname='Mini_dataset')
    write_json(RUN.parent/'Mini_dataset_6000_receipt.json',{'path':str(target),'bytes':target.stat().st_size,
        'sha256':file_hash(target),'images':6000,'claim':'provisional scheduler development pool, not verified urgency truth'})
    print('MINI DATASET READY',target,target.stat().st_size,flush=True)
'''))
    nbformat.validate(notebook)
    target=HERE/'Mini_Dataset.ipynb';nbformat.write(notebook,target)
    if target.stat().st_size>=1000000:raise ValueError('Kaggle source size exceeds limit')
    print(target,target.stat().st_size)

if __name__=='__main__':
    import sys
    build(sys.argv[1])
