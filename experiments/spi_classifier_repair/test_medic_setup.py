import hashlib
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from experiments.spi_classifier_repair.medic_training_setup import assign_roles

class SetupTests(unittest.TestCase):
    def test_roles_and_known_exposure(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows=[]
            for i in range(300):
                path=Path(tmp)/f'{i}.png';Image.new('RGB',(2,2),(i%256,i//256,0)).save(path)
                h=hashlib.sha256(path.read_bytes()).hexdigest()
                rows.append({'image_id':h,'image_sha256':h,'image_path':str(path),'group':h,'source':'medic'})
            exposed={r['image_sha256'] for r in rows[:80]}
            out,receipt=assign_roles(rows,exposed)
            self.assertEqual(receipt['actual_counts'],{'train':180,'validation':60,'test':60})
            self.assertFalse(any(r['image_sha256'] in exposed for r in out if r['split']=='test'))
            self.assertEqual([r['split'] for r in out],[r['split'] for r in assign_roles(rows,exposed)[0]])
    def test_mixed_source_rejected(self):
        with self.assertRaises(ValueError):assign_roles([{'source':'ladi'}]*300)

if __name__=='__main__':unittest.main()
