"""Versioned research header with a pre-shared, checksummed field registry.

No field-name strings or model weights are sent per message. Payload bytes are
external. CRC detects accidental corruption; transport authenticates the source.
"""
import hashlib
import math
import struct
import zlib
from .contracts import canonical, flatten_bundle

PREFIX = struct.Struct('>4sBB8s8s8sIQIH')


class HeaderCodec:
    def __init__(self, feature_names, specification_sha256, bits=8):
        self.feature_names=tuple(feature_names)
        if len(set(self.feature_names))!=len(self.feature_names) or len(self.feature_names)>65535 or bits not in (8,16):
            raise ValueError('unique field registry and 8/16-bit quantization required')
        self.bits=bits
        if not isinstance(specification_sha256,str) or len(specification_sha256)!=64:
            raise ValueError('full specification digest required')
        self.specification_sha256=specification_sha256
        self.registry_sha256=hashlib.sha256(canonical({'fields':self.feature_names,'specification':specification_sha256,'bits':bits}).encode()).hexdigest()
        self.maximum=(1<<bits)-1

    def encode(self,bundle):
        if bundle.specification_sha256!=self.specification_sha256:
            raise ValueError('bundle and registered extraction specification differ')
        flat=flatten_bundle(bundle)
        valid=[name in flat for name in self.feature_names]
        mask=bytearray((len(valid)+7)//8)
        values=[]
        for i,name in enumerate(self.feature_names):
            if valid[i]:mask[i//8]|=1<<(i%8)
            values.append(int(math.floor(flat.get(name,0)*self.maximum+.5)))
        captured=bundle.temporal['capture_time_ms'] if bundle.temporal.get('available') else 0
        delay=bundle.temporal.get('allowed_delay_ms') or 0
        if not 0<=delay<2**32:raise ValueError('allowed delay does not fit wire uint32')
        prefix=PREFIX.pack(b'BSE1',self.bits,int(bool(captured)),bytes.fromhex(self.registry_sha256[:16]),
                           bytes.fromhex(bundle.mission_version[:16]),hashlib.sha256(bundle.image_token.encode()).digest()[:8],
                           bundle.kb_version,captured,delay,len(values))
        packed=struct.pack('>'+('B' if self.bits==8 else 'H')*len(values),*values)
        body=prefix+bytes(mask)+packed
        return body+struct.pack('>I',zlib.crc32(body)&0xffffffff)

    def decode(self,wire):
        if len(wire)<PREFIX.size+4:raise ValueError('truncated header')
        magic,bits,flags,registry,mission,token,kb,captured,delay,count=PREFIX.unpack(wire[:PREFIX.size])
        expected=PREFIX.size+(count+7)//8+count*(bits//8)+4
        if magic!=b'BSE1' or bits!=self.bits or flags not in (0,1) or count!=len(self.feature_names) or registry!=bytes.fromhex(self.registry_sha256[:16]) or len(wire)!=expected:
            raise ValueError('incompatible header/registry')
        if zlib.crc32(wire[:-4])&0xffffffff!=struct.unpack('>I',wire[-4:])[0]:raise ValueError('CRC mismatch')
        if bool(flags)!=bool(captured):raise ValueError('timestamp validity mismatch')
        mask=wire[PREFIX.size:PREFIX.size+(count+7)//8]
        packed=wire[PREFIX.size+(count+7)//8:-4]
        scores=struct.unpack('>'+('B' if bits==8 else 'H')*count,packed)
        values={}
        for i,(name,value) in enumerate(zip(self.feature_names,scores)):
            if mask[i//8]&(1<<(i%8)):values[name]=value/self.maximum
            elif value:raise ValueError('missing field has nonzero wire value')
        return {'values':values,'capture_time_ms':captured or None,'allowed_delay_ms':delay or None,
                'kb_version':kb,'mission_fingerprint':mission.hex(),'token_fingerprint':token.hex()}

    def describe(self):
        n=len(self.feature_names)
        return {'schema':'BSE1','registry_sha256':self.registry_sha256,'feature_names':list(self.feature_names),
                'score_bits':n*self.bits,'availability_bits_padded':8*((n+7)//8),
                'metadata_bytes':PREFIX.size,'crc_bytes':4,
                'header_bytes':PREFIX.size+(n+7)//8+n*(self.bits//8)+4,
                'payload_bits':None,'note':'digital research header only; codec payload measured separately'}
