"""One selected spatial producer; detections are never fused or priority-weighted."""
from .scene_measurements import YOLOWorldSceneProvider
from .grounding_scene import GroundingDINOSceneProvider


def make_spatial_provider(checkpoint,*,detector='yolo_world',device='cpu'):
    if detector=='yolo_world':return YOLOWorldSceneProvider(checkpoint,device=device)
    if detector=='grounding_dino':return GroundingDINOSceneProvider(checkpoint,device=device)
    raise ValueError('detector must be yolo_world or grounding_dino; no ensemble or silent fallback')
