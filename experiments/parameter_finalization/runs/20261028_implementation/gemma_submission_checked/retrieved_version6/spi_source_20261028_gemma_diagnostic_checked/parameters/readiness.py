"""Fail closed before any new post-SPI bulk parameter cache is created."""
import hashlib
import json
from pathlib import Path
from .spi import SPIClassifier
from .spi_labels import LEVELS


def require_spi_ready(model_path, receipt_path):
    model_path, receipt_path = Path(model_path), Path(receipt_path)
    model = SPIClassifier.load(model_path)
    if model.levels != LEVELS:
        raise ValueError('cache requires all four actual SPI classes')
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    if receipt.get('runtime_promotion_allowed') is not True:
        raise ValueError('SPI readiness receipt does not permit extraction')
    checks = receipt.get('release_checks', {})
    if not checks or not all(value is True for value in checks.values()):
        raise ValueError('SPI readiness checks failed or absent')
    identity = receipt.get('spi', {})
    if identity.get('file_sha256') != hashlib.sha256(model_path.read_bytes()).hexdigest():
        raise ValueError('SPI readiness/model file identity mismatch')
    if identity.get('model_sha256') != model.artifact['sha256']:
        raise ValueError('SPI artifact identity mismatch')
    audit = receipt.get('diagnostic_review', {})
    support = receipt.get('class_support', {})
    if audit.get('accepted') is not True or audit.get('unresolved_decisive_contradictions') != 0:
        raise ValueError('SPI diagnostic review not accepted')
    for role, rows, groups in [('train', 12, 6), ('validation', 5, 3)]:
        for name in LEVELS:
            entry = support.get(role, {}).get(name, {})
            if entry.get('rows', 0) < rows or entry.get('groups', 0) < groups:
                raise ValueError('SPI class support insufficient: ' + role + '/' + name)
    return model
