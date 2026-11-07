"""Single-image inference with the frozen shared encoder and four minimal outputs."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image",required=True);parser.add_argument("--model",required=True)
    parser.add_argument("--checkpoint",required=True);parser.add_argument("--mission",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    from PIL import Image
    from .contracts import MissionSpec
    from .extractor import SenderMemory
    from .minimal_vision import MinimalVisionProvider
    from .minimal_parameters import MinimalParameterExtractor
    from .spi import SPIClassifier
    from .artifacts import write_json
    mission=MissionSpec("operator_mission",args.mission,(("mission_relevance",args.mission),))
    provider=MinimalVisionProvider(args.checkpoint,{mission.mission_id:mission},device="cpu")
    try:
        classifier=SPIClassifier.load(args.model)
        with Image.open(args.image) as image:
            result=MinimalParameterExtractor(provider,classifier).extract(image.convert("RGB"),mission,
                {"source_id":"ue01","source_numeric_id":1,"session_id":1,"message_id":1,"profile_id":1},
                SenderMemory("ue01",mission.mission_id))
        wire=result["header"].to_bytes();result["header"]=asdict(result["header"])
        result["header_bytes"]=len(wire);result["header_hex"]=wire.hex()
        write_json(args.output,result)
        print(json.dumps(result,indent=2))
    except (OSError,ValueError,RuntimeError) as error:
        write_json(args.output,{"status":"unavailable","reason":str(error)})
        raise
    finally:provider.close()


if __name__=="__main__":main()
