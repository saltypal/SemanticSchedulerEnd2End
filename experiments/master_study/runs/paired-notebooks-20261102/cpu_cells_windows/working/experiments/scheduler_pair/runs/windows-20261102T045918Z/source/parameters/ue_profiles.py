"""Operator-defined UE task bindings; no automatic importance weights or vision imports."""
from dataclasses import dataclass
import json
from pathlib import Path
from .contracts import MissionSpec
from .extractor import SenderMemory


@dataclass(frozen=True)
class UETaskBinding:
    ue_id: str
    device_type: str
    mission_id: str
    description: str


def load_ue_profiles(path=None):
    """Return mission definitions and UE bindings from one versioned operator registry.

    Device type is provenance,not a priority or scoring feature. CLIP text is the
    explicit mission.conditions clause,not a task chosen from the current image.
    """
    path=Path(path) if path is not None else Path(__file__).parent/'config/ue_task_profiles.json'
    data=json.loads(path.read_text(encoding='utf-8'))
    if set(data)!={'schema','missions','ues'} or data['schema']!='better_ue_tasks_v1':
        raise ValueError('invalid UE task registry schema')
    missions={}
    for row in data['missions']:
        mission=MissionSpec(**row)
        if mission.mission_id in missions:raise ValueError('duplicate mission identity')
        missions[mission.mission_id]=mission
    bindings=[];seen=set()
    for row in data['ues']:
        binding=UETaskBinding(**row)
        if not all(isinstance(value,str) and value.strip() for value in vars(binding).values()):
            raise ValueError('UE task binding requires nonempty strings')
        if binding.ue_id in seen:raise ValueError('duplicate UE identity')
        if binding.mission_id not in missions:raise ValueError('UE refers to an unknown mission')
        seen.add(binding.ue_id);bindings.append(binding)
    if not missions or not bindings:raise ValueError('at least one mission and UE are required')
    return missions,tuple(bindings)


def create_ue_memories(missions,bindings,capacity=64):
    """Fresh independent KB for each (UE,mission-version); caller owns serial commits."""
    output={}
    for binding in bindings:
        mission=missions[binding.mission_id];key=(binding.ue_id,mission.version)
        if key in output:raise ValueError('duplicate UE/mission KB')
        output[key]=SenderMemory(binding.ue_id,mission.mission_id,capacity)
    return output
