"""Independent priority references and class-specific delay, coverage and error metrics."""
from dataclasses import dataclass
import math
from Scheduler.spi_policy import SemanticPriority,validate_priority


TREC_LABEL_TO_SPI={'Critical':0,'High':1,'Medium':2,'Low':3}


def trecis_priority(label):
    """Explicit Low->Normal alias; it does not create an image-only reference."""
    if label not in TREC_LABEL_TO_SPI:raise ValueError('unknown official TREC-IS priority label')
    return TREC_LABEL_TO_SPI[label]


@dataclass(frozen=True)
class SPIReference:
    message_id:str
    report_id:str
    arrival:float
    reference_priority:int|None
    target_scope:str

    def __post_init__(self):
        if not self.message_id or not self.report_id or not self.target_scope:raise ValueError('reference identity/scope required')
        if not math.isfinite(self.arrival) or self.arrival<0:raise ValueError('finite nonnegative arrival required')
        if self.reference_priority is not None:validate_priority(self.reference_priority)


@dataclass(frozen=True)
class SPIOutcome:
    message_id:str
    start:float
    finish:float
    success:bool

    def __post_init__(self):
        if type(self.success) is not bool:raise ValueError('success must be boolean')
        if not math.isfinite(self.start) or not math.isfinite(self.finish) or not 0<=self.start<=self.finish:
            raise ValueError('invalid outcome times')


def evaluate_spi_outcomes(references,outcomes,*,observation_end,delay_cap,completion_budget=10):
    """Cap missed reports at a declared evaluation budget, not an invented deadline.

    Complete follow-up to arrival+cap is required. Repeats share an independently
    established report identity. Without correspondence, assign distinct units.
    Runtime predictions never define class membership for these statistics.
    """
    if not math.isfinite(observation_end) or observation_end<0 or not math.isfinite(delay_cap) or delay_cap<=0:
        raise ValueError('valid observation horizon and positive delay cap required')
    if type(completion_budget) is not int or completion_budget<=0:raise ValueError('positive completion budget required')
    references=tuple(references);outcomes=tuple(outcomes)
    lookup={r.message_id:r for r in references}
    if len(lookup)!=len(references):raise ValueError('duplicate reference message identity')
    if len({o.message_id for o in outcomes})!=len(outcomes):raise ValueError('duplicate completed attempt')
    units={}
    for r in references:
        if r.arrival>observation_end:raise ValueError('reference offered after observation end')
        if r.reference_priority is None:continue
        if r.report_id in units and units[r.report_id]['priority']!=r.reference_priority:
            raise ValueError('conflicting priority within a reference repeat group')
        u=units.setdefault(r.report_id,{'priority':r.reference_priority,'arrival':r.arrival,'scope':r.target_scope})
        u['arrival']=min(u['arrival'],r.arrival)
        if u['scope']!=r.target_scope:raise ValueError('mixed reference scope in repeat group')
    if any(u['arrival']+delay_cap>observation_end+1e-9 for u in units.values()):
        raise ValueError('insufficient follow-up for capped delay')
    previous_finish=0.
    for o in sorted(outcomes,key=lambda outcome:(outcome.start,outcome.finish,outcome.message_id)):
        if o.start<previous_finish:raise ValueError('overlapping service violates single nonpreemptive decoder')
        if o.start>observation_end:raise ValueError('service started after observation end')
        previous_finish=o.finish
    finished=[];exposed={};starts={}
    for o in outcomes:
        if o.message_id not in lookup:raise ValueError('foreign completion identity')
        r=lookup[o.message_id]
        if o.start<r.arrival:raise ValueError('service before arrival')
        if o.finish>observation_end:continue
        finished.append(o)
        if o.success and r.reference_priority is not None:
            if r.report_id not in exposed or o.finish<exposed[r.report_id]:
                exposed[r.report_id]=o.finish;starts[r.report_id]=o.start
    finished.sort(key=lambda o:(o.finish,o.start,o.message_id))
    prefix={lookup[o.message_id].report_id for o in finished[:completion_budget]
            if o.success and lookup[o.message_id].reference_priority is not None}
    by_class={}
    for priority in SemanticPriority:
        ids=[key for key,u in units.items() if u['priority']==int(priority)]
        delays=[min(exposed[key]-units[key]['arrival'],delay_cap) if key in exposed else delay_cap for key in ids]
        delivered=[key for key in ids if key in exposed]
        waiting=[starts[key]-units[key]['arrival'] for key in delivered]
        within=sum(key in exposed and exposed[key]-units[key]['arrival']<=delay_cap for key in ids)
        n=len(ids)
        by_class[priority.name.lower()]={'offered_distinct_reports':n,'successfully_exposed':len(delivered),
            'mean_capped_completion_delay':sum(delays)/n if n else None,
            'exposure_within_evaluation_budget':within/n if n else None,
            'undelivered_fraction':1-len(delivered)/n if n else None,
            'delivered_only_mean_wait':sum(waiting)/len(waiting) if waiting else None,
            'distinct_recall_at_completions':sum(key in prefix for key in ids)/n
                if n and len(finished)>=completion_budget else None}
    return {'reference_scope':sorted({r.target_scope for r in references}),
        'delay_cap':delay_cap,'cap_is_evaluation_budget_not_operational_deadline':True,
        'completion_budget':completion_budget,'completed_attempts':len(finished),
        'unknown_reference_messages':sum(r.reference_priority is None for r in references),
        'by_reference_class':by_class,
        'primary':'critical.mean_capped_completion_delay; report critical exposure and all class outcomes alongside it'}


def reference_choice_violation(candidates,selected_id,references):
    """One dispatch decision only; later critical arrivals do not create inversions."""
    ids={c.message_id for c in candidates}
    if selected_id not in ids:raise ValueError('selected message not ready')
    values={key:validate_priority(references[key]) for key in ids if references.get(key) is not None}
    if set(values)!=ids:return {'defined':False,'violation':None,'reason':'missing reference class'}
    return {'defined':True,'violation':values[selected_id]>min(values.values()),
            'selected_reference_spi':values[selected_id],'best_ready_reference_spi':min(values.values())}
