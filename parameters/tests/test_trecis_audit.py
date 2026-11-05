"""Reference cases for data grain and TREC SGML parsing."""
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from parameters.audit_trecis import audit, topic_records


class TrecisAuditTests(unittest.TestCase):
    def test_missing_data_cannot_produce_an_empty_successful_audit(self):
        with TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                audit(Path(folder))

    def test_topics_are_concatenated_blocks_and_preserve_text(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "topics.xml"
            path.write_text('<top><num>TRECIS-CTIT-H-Training-001</num><title>A & B</title></top>'
                            '<top><num>TRECIS-CTIT-H-122</num><title>Second</title></top>', encoding="utf8")
            records = topic_records(path)
            self.assertEqual([r["number"] for r in records], [1, 122])
            self.assertEqual(records[0]["title"], "A & B")

    def test_merged_parent_is_not_an_extra_vote_and_copies_do_not_cross_grain(self):
        def row(event, priority, assessor):
            return {"eventID": event, "postID": "10", "postCategories": ["Screenshot"],
                    "postPriority": priority, "postText": "same", "annotator": assessor,
                    "multipleJudgements": None}
        with TemporaryDirectory() as folder:
            run = Path(folder)
            raw = run / "raw"
            raw.mkdir()
            low = row("eventA", "Low", "person1")
            high = row("eventA", "High", "person2")
            parent = {**high, "multipleJudgements": [low, high]}
            other_event = row("eventB", "High", "person3")
            events = [{"eventid": "eventA", "tweets": [parent, low]},
                      {"eventid": "eventB", "tweets": [other_event]}]
            with gzip.open(raw / "annotations_fixture.json.gz", "wt", encoding="utf8") as handle:
                json.dump({"metadata": {}, "events": events}, handle)
            (raw / "information_types.json").write_text('{"informationTypes": []}', encoding="utf8")
            result = audit(run)
            self.assertEqual(result["combined"]["unique_post_ids"], 1)
            self.assertEqual(result["combined"]["unique_event_post_pairs"], 2)
            self.assertEqual(result["combined"]["unique_leaf_records"], 3)
            self.assertEqual(result["combined"]["posts_with_conflicting_leaf_priority"], 1)
            self.assertEqual(result["candidate_event_components"]["count"], 1)
            self.assertEqual(result["image_annotation_proxy"]["unique_posts_with_any"], 1)

    def test_unclosed_topic_is_an_error(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "broken.xml"
            path.write_text('<top><num>001</num>', encoding="utf8")
            with self.assertRaises(ValueError):
                topic_records(path)


if __name__ == "__main__":
    unittest.main()
