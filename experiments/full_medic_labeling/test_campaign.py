import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np

from experiments.full_medic_labeling.campaign import LabelStore
from experiments.full_medic_labeling.teacher import SYSTEM_PROMPT,PROMPT_SHA256


class LabelCampaignTests(unittest.TestCase):
    def item(self,identity,sha,split='train'):
        return {'item_id':identity,'image_sha256':sha,'source_split':split,
                'source_shard':'test_fixture.parquet','shard_row':0,'source_metadata':{}}

    def test_durable_resume_aliases_and_unknown_not_low(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);store=LabelStore(d/'work.sqlite',{'prompt':PROMPT_SHA256})
            result={'available':True,'level':'High','status':'complete'}
            store.add(self.item('one','sha'),result,np.ones(512,dtype=np.float32))
            store.add(self.item('two','sha','dev'),result,np.ones(512,dtype=np.float32))
            store.add(self.item('three','unknown'),{'available':False,'level':None,'status':'failed'})
            progress=store.export(d/'export')
            self.assertEqual(progress['attempted_rows'],3)
            self.assertEqual(progress['unique_image_hashes'],2)
            self.assertEqual(progress['class_counts'],{'High':2,'unavailable':1})
            self.assertFalse(progress['coverage_complete'])
            store.db.close()
            resumed=LabelStore(d/'export/full_medic_checkpoint.sqlite',{'prompt':PROMPT_SHA256})
            self.assertTrue(resumed.has_item('two'))
            self.assertEqual(resumed.prediction('sha')[0],result)
            self.assertIsNone(resumed.prediction('unknown')[0]['level'])
            resumed.db.close()
            with self.assertRaises(ValueError):
                LabelStore(d/'export/full_medic_checkpoint.sqlite',{'prompt':'different'})

    def test_system_rubric(self):
        self.assertIn('Critical:',SYSTEM_PROMPT)
        self.assertIn('Do not make classes equally frequent',SYSTEM_PROMPT)
        self.assertIn('level=null',SYSTEM_PROMPT)


if __name__=='__main__':unittest.main()
