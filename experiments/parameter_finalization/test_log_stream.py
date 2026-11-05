"""Console reporter compatibility while preserving incremental log output."""
import io
from parameters.artifacts import Tee

def test_reporter_terminal_detection_and_nested_log_capture():
    terminal=io.StringIO();first=io.StringIO();second=io.StringIO()
    stream=Tee(Tee(terminal,first),second)
    assert stream.isatty() is False
    assert stream.encoding == terminal.encoding
    assert stream.write('model report')==12
    stream.flush()
    assert terminal.getvalue()==first.getvalue()==second.getvalue()=='model report'
    second.close()
    stream.write(' still live')
    assert terminal.getvalue().endswith(' still live')
