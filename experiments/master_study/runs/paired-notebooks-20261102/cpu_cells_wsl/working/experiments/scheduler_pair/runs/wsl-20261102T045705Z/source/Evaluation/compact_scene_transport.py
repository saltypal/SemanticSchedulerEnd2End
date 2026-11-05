"""Lossless transport of scene-array-v2 measurements under a preinstalled profile.
No raw geometry recovery, learned field removal, or authentication is implied.
"""
import struct


class CompactSceneTransport:
    VERSION = 3
    PREFIX = struct.Struct('<BBH')

    def __init__(self, profile_id, registry_sha256, producer_sha256, count_applicable):
        if type(profile_id) is not int or not 0 <= profile_id <= 65535:
            raise ValueError('uint16 configured profile ID required')
        self.profile_id = profile_id
        self.registry = bytes.fromhex(registry_sha256)
        self.producer = bytes.fromhex(producer_sha256)
        self.applicable = tuple(count_applicable)
        if len(self.registry) != 32 or len(self.producer) != 32 or len(self.applicable) != 93:
            raise ValueError('full hashes and exact93-class profile required')
        if any(type(x) is not bool for x in self.applicable):
            raise ValueError('boolean applicability required')

    def _validate(self, wire):
        if len(wire) != 369 or wire[0] != 2 or wire[1] & ~3:
            raise ValueError('invalid source scene version/size/flags')
        if wire[2:34] != self.registry or wire[34:66] != self.producer:
            raise ValueError('scene producer/vocabulary profile mismatch')
        counts = struct.unpack_from('<93H', wire, 66)
        areas = wire[252:345]
        cm = int.from_bytes(wire[345:357], 'little')
        am = int.from_bytes(wire[357:369], 'little')
        if cm >> 93 or am >> 93:
            raise ValueError('nonzero mask padding')
        for i, (n,a) in enumerate(zip(counts,areas)):
            c, r = bool(cm & (1 << i)), bool(am & (1 << i))
            if c and not self.applicable[i] or not c and n or not r and a:
                raise ValueError('invalid count applicability or unavailable values')
            if not wire[1] & 1 and (c or r or n or a):
                raise ValueError('failed detector exposes evidence')
        return counts, areas

    def encode(self, wire):
        counts, areas = self._validate(wire)
        dense = self.PREFIX.pack(3, wire[1], self.profile_id) + wire[66:]
        entries = [(i,n,a) for i,(n,a) in enumerate(zip(counts,areas)) if n or a]
        sparse = (self.PREFIX.pack(3,wire[1] | 4,self.profile_id) + bytes([len(entries)])
                  + wire[345:] + b''.join(struct.pack('<BHB',*e) for e in entries))
        return sparse if len(sparse) < len(dense) else dense

    def decode(self, data):
        if len(data) < 4:
            raise ValueError('truncated compact prefix')
        version, flags, profile = self.PREFIX.unpack_from(data)
        if version != 3 or flags & ~7 or profile != self.profile_id:
            raise ValueError('unknown compact profile/version/flags')
        if flags & 4:
            if len(data) < 29 or data[4] > 93 or len(data) != 29 + 4*data[4]:
                raise ValueError('invalid sparse length/count')
            counts, areas, last = [0]*93, bytearray(93), -1
            for k in range(data[4]):
                i,n,a = struct.unpack_from('<BHB',data,29+4*k)
                if i <= last or i >= 93 or not (n or a):
                    raise ValueError('unsorted/duplicate/invalid sparse entry')
                counts[i],areas[i],last = n,a,i
            body = struct.pack('<93H',*counts) + bytes(areas) + data[5:29]
        else:
            if len(data) != 307:
                raise ValueError('invalid dense length')
            body = data[4:]
        wire = bytes([2,flags & 3]) + self.registry + self.producer + body
        self._validate(wire)
        return wire
