"""Trial scene transport retaining the five project objects and eight regions."""

import struct


class NoCocoSceneTransport:
    """Use a separate wire version and expand omitted slots as unavailable.

    Both endpoints configure the same legacy producer and registry binding.
    Expansion keeps the existing receiver's feature shape and slot identities.
    """

    VERSION = 4
    VOCABULARY_SHA256 = "37181cef81ba6f9c777ab05fefe182a50dedaeda8454c7ad9216ff17e1eb91c0"
    PREFIX = struct.Struct("<BBH")
    DENSE = struct.Struct("<5H13sBH")
    ENTRY = struct.Struct("<BHB")
    MASKS = struct.Struct("<BH")
    COCO_SLOTS = 80
    RETAINED_SLOTS = 13
    OBJECT_SLOTS = 5

    def __init__(self, legacy):
        expected = (True,) * 85 + (False,) * 8
        if legacy.registry.hex() != self.VOCABULARY_SHA256 or legacy.applicable != expected:
            raise ValueError("expected the ordered COCO80 + project5 + region8 profile")
        self.legacy = legacy

    def encode(self, wire):
        counts, areas = self.legacy._validate(wire)
        counts = counts[self.COCO_SLOTS:self.COCO_SLOTS + self.OBJECT_SLOTS]
        areas = areas[self.COCO_SLOTS:]
        cm = int.from_bytes(wire[345:357], "little") >> self.COCO_SLOTS
        am = int.from_bytes(wire[357:369], "little") >> self.COCO_SLOTS
        flags, profile = wire[1], self.legacy.profile_id
        dense = self.PREFIX.pack(self.VERSION, flags, profile) + self.DENSE.pack(*counts, areas, cm, am)
        padded_counts = counts + (0,) * (self.RETAINED_SLOTS - self.OBJECT_SLOTS)
        entries = [self.ENTRY.pack(i, count, area)
                   for i, (count, area) in enumerate(zip(padded_counts, areas)) if count or area]
        sparse = (self.PREFIX.pack(self.VERSION, flags | 4, profile) + bytes([len(entries)])
                  + self.MASKS.pack(cm, am) + b"".join(entries))
        return sparse if len(sparse) < len(dense) else dense

    def decode(self, data):
        if len(data) < self.PREFIX.size:
            raise ValueError("truncated scene prefix")
        version, flags, profile = self.PREFIX.unpack_from(data)
        if version != self.VERSION or flags & ~7 or profile != self.legacy.profile_id:
            raise ValueError("unknown no-COCO scene version, flags or profile")
        if flags & 4:
            offset = self.PREFIX.size + 1 + self.MASKS.size
            if len(data) < offset:
                raise ValueError("truncated sparse scene")
            size = data[self.PREFIX.size]
            if size > self.RETAINED_SLOTS or len(data) != offset + size * self.ENTRY.size:
                raise ValueError("invalid sparse scene length")
            cm, am = self.MASKS.unpack_from(data, self.PREFIX.size + 1)
            counts, areas, previous = [0] * self.OBJECT_SLOTS, bytearray(self.RETAINED_SLOTS), -1
            for i, count, area in self.ENTRY.iter_unpack(data[offset:]):
                if not previous < i < self.RETAINED_SLOTS or not (count or area):
                    raise ValueError("invalid or duplicate sparse scene slot")
                if i < self.OBJECT_SLOTS:
                    counts[i] = count
                elif count:
                    raise ValueError("region count is inapplicable")
                areas[i], previous = area, i
        else:
            if len(data) != self.PREFIX.size + self.DENSE.size:
                raise ValueError("invalid dense scene length")
            *counts, areas, cm, am = self.DENSE.unpack_from(data, self.PREFIX.size)
        if cm >> self.OBJECT_SLOTS or am >> self.RETAINED_SLOTS:
            raise ValueError("nonzero scene mask padding")
        wire = (bytes([2, flags & 3]) + self.legacy.registry + self.legacy.producer
                + bytes(2 * self.COCO_SLOTS) + struct.pack("<5H", *counts) + bytes(16)
                + bytes(self.COCO_SLOTS) + areas
                + (cm << self.COCO_SLOTS).to_bytes(12, "little")
                + (am << self.COCO_SLOTS).to_bytes(12, "little"))
        self.legacy._validate(wire)
        return wire
