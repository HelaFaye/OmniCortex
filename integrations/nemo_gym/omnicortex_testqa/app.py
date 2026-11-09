"""NeMo Gym resources server: OmniCortex's TestQA grading as a reward (scripts/gym_bridge.py).

Copy this folder to Gym's resources_servers/omnicortex_testqa and set OMNICORTEX_ROOT to an OmniCortex checkout.
Correct answers earn 1, declines `abstain_reward`, confident mistakes `wrong_reward`: the policy is not paid for
guessing. With `studio_url` (and the model's classifier set up there), correct answers given at high hallucination
risk earn `risk_weight` x risk less.
"""
import os
import sys
from pathlib import Path
from typing import Any, Optional

from nemo_gym.server_runtime import (BaseResourcesServerConfig, BaseVerifyRequest, BaseVerifyResponse,
                                     SimpleResourcesServer, observe_programmatic_grading)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint

sys.path.insert(0, str(Path(os.environ.get("OMNICORTEX_ROOT", Path(__file__).resolve().parents[3])) / "scripts"))
import gym_bridge  # noqa: E402


class OmniCortexTestQAConfig(BaseResourcesServerConfig):
    abstain_reward: float = 0.25
    wrong_reward: float = -1.0
    risk_weight: float = 0.0
    studio_url: Optional[str] = None
    studio_model: Optional[str] = None


class OmniCortexVerifyRequest(BaseVerifyRequest):
    omnicortex_task: dict[str, Any]


class OmniCortexVerifyResponse(BaseVerifyResponse):
    verdict: str
    risk: Optional[float] = None


class OmniCortexTestQAServer(SimpleResourcesServer):
    config: OmniCortexTestQAConfig
    checkpoint_mode = "stateless"

    def _score(self, body: OmniCortexVerifyRequest) -> OmniCortexVerifyResponse:
        text = gym_bridge.response_text(body.response)
        risk = None
        if self.config.studio_url and self.config.risk_weight:
            msgs = [m if isinstance(m, dict) else m.model_dump() for m in body.responses_create_params.input]
            risk = gym_bridge.risk_of(self.config.studio_url, self.config.studio_model or "", msgs, text)
        r = gym_bridge.reward(body.omnicortex_task, text, risk, {"abstain": self.config.abstain_reward,
                                                                 "wrong": self.config.wrong_reward,
                                                                 "risk_weight": self.config.risk_weight})
        return OmniCortexVerifyResponse(**body.model_dump(), reward=r["reward"], verdict=r["verdict"], risk=r.get("risk"))

    async def verify(self, body: OmniCortexVerifyRequest) -> OmniCortexVerifyResponse:
        return observe_programmatic_grading(lambda: self._score(body))


if __name__ == "__main__":
    OmniCortexTestQAServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = OmniCortexTestQAServer.run_webserver()  # noqa: F401
