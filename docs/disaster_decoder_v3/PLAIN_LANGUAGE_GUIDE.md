# Plain-language guide

The receiver gets an image's complete compressed digital representation and a small header. The proposed experiment asks whether the header helps the receiver choose which already-arrived image representation matters most for a task. It does not yet demonstrate that it works, and it does not rank partially received packets.

The header carries eight optional clues: task match (`R`), how rare the observed meaning is in this transmitter's recent history (`S`), confidence in one important claim (`Q`), task-specific importance (`I`), trusted recency (`F`), visible target area (`P`), scene complexity (`C`), and difference from recent images (`N`). A missing clue is marked unavailable, not silently treated as zero. These are evidence features, not guaranteed truths.

Building, road, and structural-flooding tasks have distinct profile IDs and label weights. Frozen image models provide candidate signals; LADI-small is only a reference. Qwen annotations are silver labels, so their results must not be mixed with source-grounded labels. A designed observation set is not proof that all real damage was observed, and two different photos do not prove two different incidents.

The code verified so far checks the parameter math and 32-byte header contract with synthetic tests. It has not established image-model accuracy, trained a ranker, compared baselines, or run the held-out test. The next meaningful checkpoint is a reproducible extraction cache with source/model/split provenance, followed by training and matched comparisons.
