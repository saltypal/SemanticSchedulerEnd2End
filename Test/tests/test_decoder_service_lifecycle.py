"""Exercise the real receiver queue/controller with serialized headers and opaque payloads."""
from dataclasses import replace
import pytest
from Archived.owners_before_simplification_20261027.SemParam.decoder_parameters import ParameterHeader
from Evaluation.decoder_queue import ReceivedMessage
from Evaluation.decoder_service import DecoderService
from Scheduler.decoder_scheduler import SemanticScheduler
from Scheduler.contextual_decoder_scheduler import StaticPolicyAdapter


def message(token, source="ue", arrival=0., sequence=0):
    header=ParameterHeader(1001,token,0,0,(0.,)*8,(False,)*8).to_bytes()
    return ReceivedMessage(source,"session",token,header,object(),True,arrival,sequence)


def service(kind="fifo"):
    return DecoderService(StaticPolicyAdapter(SemanticScheduler(kind,profile_id=1001)),1001)


def test_serialized_ingress_dispatch_opaque_payload_and_success():
    receiver=service();job=message(1)
    receiver.enqueue(job)
    dispatched=receiver.dispatch(0.)
    assert dispatched.message.payload_ref is job.payload_ref
    assert not hasattr(dispatched.header_snapshot,"payload_ref")
    assert receiver.dispatch(.5) is None
    result=receiver.complete(job.message_id,"succeeded",1.)
    assert result.service_seconds==1.
    assert len(receiver.history)==1 and receiver.in_flight is None
    assert receiver.dispatch(1.) is None


def test_concurrent_arrival_waits_for_nonpreemptive_completion():
    receiver=service();first=message(1);second=message(2,"other",.5,1)
    receiver.enqueue(first);receiver.dispatch(0.)
    receiver.enqueue(second)
    assert receiver.dispatch(.5) is None
    receiver.complete(first.message_id,"succeeded",1.)
    assert receiver.dispatch(1.).message.message_id==second.message_id


def test_failure_exact_once_and_foreign_completion_rejected():
    receiver=service();job=message(1);receiver.enqueue(job);receiver.dispatch(0.)
    with pytest.raises(ValueError):
        receiver.complete(message(1,"foreign").message_id,"succeeded",1.)
    receiver.complete(job.message_id,"failed",1.)
    assert not receiver.history and len(receiver.completions)==1
    with pytest.raises(ValueError):
        receiver.complete(job.message_id,"succeeded",1.)
    assert receiver.dispatch(1.) is None  # No retry.


@pytest.mark.parametrize("fault",["incomplete","duplicate","crc","token"])
def test_ingress_rejects_invalid_messages(fault):
    receiver=service();job=message(1)
    if fault=="incomplete":job=replace(job,complete=False)
    if fault=="duplicate":receiver.enqueue(job)
    if fault=="crc":job=replace(job,header_bytes=job.header_bytes[:-1]+bytes([job.header_bytes[-1]^1]))
    if fault=="token":job=replace(job,message_token=2)
    with pytest.raises(ValueError):
        receiver.enqueue(job)


def test_future_empty_reset_and_monotonic_clock():
    receiver=service();job=message(1,arrival=2.)
    receiver.enqueue(job);assert receiver.dispatch(0.) is None
    receiver.dispatch(2.)
    with pytest.raises(RuntimeError):receiver.reset()
    with pytest.raises(ValueError):receiver.complete(job.message_id,"succeeded",1.)
    receiver.complete(job.message_id,"succeeded",3.)
    with pytest.raises(ValueError):receiver.dispatch(2.)
    receiver.reset()
    assert len(receiver)==0 and not receiver.history and not receiver.completions
    assert receiver.dispatch(0.) is None


def test_round_robin_rotates_even_when_all_semantic_fields_missing():
    receiver=service("round_robin")
    jobs=[message(1,"a"),message(2,"a",sequence=1),message(3,"b",sequence=2)]
    for job in jobs:receiver.enqueue(job)
    sources=[]
    for now in range(3):
        job=receiver.dispatch(float(now));sources.append(job.message.source_id)
        receiver.complete(job.message.message_id,"succeeded",float(now+1))
    assert sources==["a","b","a"]
