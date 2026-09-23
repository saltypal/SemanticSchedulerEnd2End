"""Summarize availability and provenance without treating proxies as labels."""
def evaluate(bundles):
    fields = {}
    for bundle in bundles:
        for name, value in bundle.get('values', {}).items():
            summary = fields.setdefault(name, {'observations':0,'available':0,'unavailable':0})
            summary['observations'] += 1
            summary['available' if value['available'] else 'unavailable'] += 1
    return {'images':len(bundles),'fields':fields,'accuracy':None,
            'accuracy_reason':'No independent ground-truth labels supplied'}
