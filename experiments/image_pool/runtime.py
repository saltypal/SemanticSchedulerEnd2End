"""Separate image-dependent SPI cache and source-preserving scheduler exports."""
import hashlib
import json
import sqlite3
from pathlib import Path
from experiments.scene_array_cache.cache import write_json, file_hash, code_hash


def cache_student(rows, student, output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    identity={'student_bundle_sha256':student.identity_hash,
              'cache_source_sha256':code_hash(__file__),
              'inputs':[(r['image_id'],r['image_sha256']) for r in rows]}
    signature=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    db=sqlite3.connect(output/'spi_cache.sqlite')
    db.executescript('CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT);'
                     'CREATE TABLE IF NOT EXISTS outputs(id TEXT PRIMARY KEY,result TEXT,checksum TEXT);'
                     'CREATE TABLE IF NOT EXISTS attempts(id TEXT,result TEXT,checksum TEXT);')
    old=db.execute("SELECT value FROM meta WHERE key='identity'").fetchone()
    if old and old[0]!=signature:db.close();raise ValueError('SPI cache identity differs')
    with db:db.execute('INSERT OR IGNORE INTO meta VALUES(?,?)',('identity',signature))
    write_json(output/'identity.json',identity)
    try:
        for index,row in enumerate(rows):
            if file_hash(row['image_path'])!=row['image_sha256']:raise ValueError('SPI source bytes changed')
            old=db.execute('SELECT result,checksum FROM outputs WHERE id=?',(row['image_id'],)).fetchone()
            if old:
                if hashlib.sha256(old[0].encode()).hexdigest()!=old[1]:raise ValueError('SPI cache corrupt')
                if json.loads(old[0])['spi'].get('available'):continue
                with db:db.execute('INSERT INTO attempts VALUES(?,?,?)',(row['image_id'],*old))
            try:result=student.predict(row['image_path'])
            except Exception as error:
                result={'available':False,'status':'failed','reason':type(error).__name__+': '+str(error),
                        'student_bundle_sha256':student.identity_hash}
            record={'image_id':row['image_id'],'image_sha256':row['image_sha256'],'spi':result}
            text=json.dumps(record,sort_keys=True,allow_nan=False);checksum=hashlib.sha256(text.encode()).hexdigest()
            with db:db.execute('INSERT OR REPLACE INTO outputs VALUES(?,?,?)',(row['image_id'],text,checksum))
            if (index+1)%25==0: print('SPI CACHE',index+1,'/',len(rows),flush=True)
    finally:
        with (output/'spi_predictions.jsonl.partial').open('w',encoding='utf8') as stream:
            for text,checksum in db.execute('SELECT result,checksum FROM outputs ORDER BY id'):
                if hashlib.sha256(text.encode()).hexdigest()!=checksum:raise ValueError('cannot export corrupt SPI row')
                stream.write(text+'\n')
        (output/'spi_predictions.jsonl.partial').replace(output/'spi_predictions.jsonl')
        db.close()
    return output/'spi_predictions.jsonl'


def join_runtime(rows, scene_jsonl, spi_jsonl, output):
    """No evaluator labels in the runtime cache. Missing mission/history stays absent."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    scene={r['image_id']:r for r in map(json.loads,Path(scene_jsonl).read_text().splitlines())}
    spi={r['image_id']:r for r in map(json.loads,Path(spi_jsonl).read_text().splitlines())}
    expected={r['image_id'] for r in rows}
    if set(scene)!=expected or set(spi)!=expected:raise ValueError('cache ID coverage mismatch')
    complete=0
    with (output/'image_parameters.jsonl.partial').open('w',encoding='utf8') as stream:
        for r in rows:
            i=r['image_id']
            if scene[i]['image_sha256']!=r['image_sha256'] or spi[i]['image_sha256']!=r['image_sha256']:
                raise ValueError('cache byte identity mismatch')
            ok=scene[i]['status']=='ok' and spi[i]['spi']['available']
            complete+=ok
            record={**r,'scene':scene[i]['scene'],'scene_audit':scene[i].get('audit'),
                'scene_status':scene[i]['status'],'scene_wire_hex':scene[i]['wire_hex'],
                'scene_timing_ms':scene[i].get('timing'),'spi':spi[i]['spi'],
                'cache_complete':ok,'mission_relevance_available':False,'novelty_available':False,
                'runtime_fields':'simulator computes relevance/novelty from mission and causalUE stream'}
            # These must stay outside scheduler runtime observations even if an input manifest contains them.
            for key in ('native_annotations','teacher_spi','priority_label','label','ground_truth'):
                if key in record:raise ValueError('evaluator field crossed runtime boundary')
            stream.write(json.dumps(record,allow_nan=False)+'\n')
    (output/'image_parameters.jsonl.partial').replace(output/'image_parameters.jsonl')
    summary={'expected':len(rows),'complete':complete,'failed_or_partial':len(rows)-complete,
             'all_complete':complete==len(rows),'SPI_semantic_accuracy':'not established on these domains',
             'relevance_novelty':'deliberately not cached as fixedimage values'}
    write_json(output/'completion.json',summary)
    return summary


def profile_spi(rows,bundle,output,devices=('cpu','cuda:0')):
    import gc,time,numpy as np,psutil,torch
    from experiments.image_pool.student import Student
    from experiments.full_medic_labeling.costs import operator_flops
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    fixed=[r for r in rows if r['split']=='train'][:10]
    if len(fixed)!=10:raise ValueError('ten fixed development images needed for SPI profiling')
    for device in devices:
        student=Student(bundle,device)
        for _ in range(10):student.predict(fixed[0]['image_path'])
        measured=[]
        for r in fixed:
            for _ in range(3 if device=='cpu' else 10):measured.append(student.predict(r['image_path'])['timing_ms'])
        if device.startswith('cuda'):torch.cuda.reset_peak_memory_stats(device)
        student.predict(fixed[0]['image_path']);student.sync()
        memory={'rss_bytes':psutil.Process().memory_info().rss,
                'cuda_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.startswith('cuda') else None,
                'cuda_reserved_bytes':torch.cuda.max_memory_reserved(device) if device.startswith('cuda') else None}
        flops=operator_flops(lambda:student.predict(fixed[0]['image_path']),device=device,batch=1,image_size=224,
            scope='supported neural operators only; sklearn component uncounted; partial')
        write_json(output/(device.replace(':','_')+'.json'),{'device':device,'image_ids':[r['image_id'] for r in fixed],
            'warmups':10,'calls':len(measured),'samples_ms':measured,
            'summary':{k:{'p50_ms':float(np.median([r[k] for r in measured])),
                          'p95_ms':float(np.percentile([r[k] for r in measured],95))} for k in measured[0]},
            'memory':memory,'encoder_model_load_seconds':student.encoder_load_seconds,
            'head_load_seconds':student.head_load_seconds,'load_scope':'freshobject;OS/downloadcachemaybewarm',
            'flops':flops,'scope':'rawimage to SPI, no scene inference or simulator calculations'})
        del student;gc.collect()
        if device.startswith('cuda'):torch.cuda.empty_cache()
