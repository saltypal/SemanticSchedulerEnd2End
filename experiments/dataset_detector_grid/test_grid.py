import unittest
from detector_grid import configurations
from cloud_grid import acquire_medic

class GridTests(unittest.TestCase):
    def test_sizes_unique_and_baselines(self):
        for name,count in [('yolo_world',20),('grounding_dino',56)]:
            grid=configurations(name)
            self.assertEqual(len(grid),count)
            self.assertEqual(len({tuple(sorted(r.items())) for r in grid}),count)
            self.assertEqual({r['max_detections'] for r in grid},{100,300,600})
    def test_invalid_model(self):
        with self.assertRaises(ValueError):configurations('unknown')
    def test_individual_acquisition_stops_at_300(self):
        import io,json,tempfile
        from PIL import Image
        class Budget:
            downloaded=0
            def get(self,url,**kwargs):
                if '/size?' in url:return json.dumps({'size':{'splits':[{'split':'train','num_rows':400}]}}).encode()
                if '/rows?' in url:
                    from urllib.parse import urlparse,parse_qs
                    offset=int(parse_qs(urlparse(url).query)['offset'][0])
                    return json.dumps({'rows':[{'row_idx':i,'row':{'image':{'src':f'image:{i}'},'informative':1}} for i in range(offset,offset+100)]}).encode()
                i=int(url.split(':')[1]);buffer=io.BytesIO()
                Image.new('RGB',(2,2),(i%256,i//256,0)).save(buffer,format='PNG')
                data=buffer.getvalue();self.downloaded+=len(data);return data
        with tempfile.TemporaryDirectory() as output:
            rows,failures,source=acquire_medic(output,Budget())
            self.assertEqual(len(rows),300);self.assertEqual(len({r['image_id'] for r in rows}),300)
            self.assertEqual(failures,[]);self.assertTrue(all(r['source']=='medic' for r in rows))

if __name__=='__main__':unittest.main()
