"""Focused protocol regression without importing optional vision dependencies."""
import importlib.util
from pathlib import Path
import struct
import unittest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('compact_scene_transport',ROOT/'Evaluation/compact_scene_transport.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class CompactSceneTests(unittest.TestCase):
    def setUp(self):
        self.c=module.CompactSceneTransport(17,'00'*32,'11'*32,[True]*85+[False]*8)

    def wire(self,counts=None,areas=None,ok=True):
        counts=counts if counts is not None else [0]*93
        areas=areas if areas is not None else [0]*93
        cm=(1<<85)-1 if ok else 0
        am=(1<<93)-1 if ok else 0
        return (bytes([2,int(ok)])+bytes(32)+bytes.fromhex('11'*32)+struct.pack('<93H',*counts)
                +bytes(areas)+cm.to_bytes(12,'little')+am.to_bytes(12,'little'))

    def test_sparse_exact(self):
        c=[0]*93;c[0]=65535;a=[0]*93;a[90]=17
        w=self.wire(c,a);x=self.c.encode(w)
        self.assertEqual(len(x),37);self.assertEqual(self.c.decode(x),w)

    def test_dense_exact(self):
        w=self.wire([1]*85+[0]*8,[20]*93);x=self.c.encode(w)
        self.assertEqual(len(x),307);self.assertEqual(self.c.decode(x),w)

    def test_zero_vs_failed(self):
        a,b=self.wire(),self.wire(ok=False)
        self.assertEqual(len(self.c.encode(a)),29)
        self.assertNotEqual(self.c.encode(a),self.c.encode(b))
        self.assertEqual(self.c.decode(self.c.encode(b)),b)

    def test_profile_and_versions(self):
        x=self.c.encode(self.wire())
        for malformed in [x[:1],bytes([4])+x[1:],x[:2]+bytes([18,0])+x[4:],bytes([3,8])+x[2:]]:
            with self.assertRaises(ValueError):self.c.decode(malformed)
        with self.assertRaises(ValueError):self.c.encode(self.wire()[:50])
        w=bytearray(self.wire());w[2]=1
        with self.assertRaises(ValueError):self.c.encode(bytes(w))

    def test_unavailable_and_regions(self):
        c=[0]*93;c[92]=1
        with self.assertRaises(ValueError):self.c.encode(self.wire(c))
        w=bytearray(self.wire());w[356]|=128
        with self.assertRaises(ValueError):self.c.encode(bytes(w))
        w=bytearray(self.wire(ok=False));w[66]=1
        with self.assertRaises(ValueError):self.c.encode(bytes(w))

    def test_bad_sparse_entries(self):
        head=struct.pack('<BBH',3,5,17)
        masks=self.wire()[345:]
        for entries in [[(0,1,0),(0,1,0)],[(93,1,0)],[(0,0,0)]]:
            x=head+bytes([len(entries)])+masks+b''.join(struct.pack('<BHB',*e) for e in entries)
            with self.assertRaises(ValueError):self.c.decode(x)


if __name__=='__main__':unittest.main()
