# Earlier simulation contracts

`contracts.py` and `simulation.py` implement the original slotted fixtures.
`semantic_contracts.py`, `streams.py`, `requests.py` and `event_simulation.py`
implement the earlier five-UE semantic experiment. These stay separate from
the device-independent runtime in `INFRA/latent`.

Existing DQN evaluations retain their observation shapes and policy rules.
Changing source ownership paths does not change their experimental meaning.
