from INFRA.errors import ComponentUnavailableError


class _UnavailableDeploymentAdapter:
    component = "Deployment adapter"

    def __init__(self) -> None:
        raise ComponentUnavailableError(self.component, "Stage 1A", "Deployment integration is explicitly deferred.")


class NS3Adapter(_UnavailableDeploymentAdapter):
    component = "NS-3 adapter"


class SionnaSystemAdapter(_UnavailableDeploymentAdapter):
    component = "Sionna system-level adapter"


class ORANAdapter(_UnavailableDeploymentAdapter):
    component = "O-RAN adapter"
