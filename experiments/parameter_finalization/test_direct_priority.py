import pytest
from parameters.spi_direct_priority import parse_class,assess_direct


@pytest.mark.parametrize('raw,expected',[('CRITICAL','Critical'),('HIGH','High'),('MEDIUM','Medium'),('LOW','Low'),(' low\n','Low')])
def test_exact_class(raw,expected):
    result,_=parse_class(raw)
    assert result=={'level':expected,'available':True,'reason':None}


@pytest.mark.parametrize('raw',['UNKNOWN','NULL','high because damage','{"level":"High"}','Critical or High',''])
def test_no_silent_label_from_invalid_response(raw):
    with pytest.raises(ValueError):parse_class(raw)


def test_direct_assessor_uses_image_and_no_boolean_schema(tmp_path):
    class Reviewer:
        def review_structured(self,paths,payload,prompt,parser,*args,**kwargs):
            assert paths==['image.png'] and set(payload)=={'instruction'}
            assert 'You are a disaster and emergency management officer' in prompt
            assert 'action_needed' not in prompt
            assert kwargs['text_only_diagnostic'] is False
            parsed,_=parser('MEDIUM')
            return {'status':'complete','parsed':parsed}
    result=assess_direct(Reviewer(),['image.png'],{'instruction':'label'},'case',tmp_path)
    assert result['level']=='Medium'
