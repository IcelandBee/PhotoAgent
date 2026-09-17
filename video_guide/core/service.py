"""Python entrypoint backed by the compiled Guidance subgraph."""
from dataclasses import asdict
from pathlib import Path
from .models import GuideInput, GuideResult
from .graph import build_guide_graph

class GuideService:
    def __init__(self, backend=None, *, checkpointer=None):
        self.graph = build_guide_graph(backend, checkpointer=checkpointer)

    def run(self, inputs: GuideInput, session_directory: Path, *, config=None):
        values = {k: str(v) if isinstance(v, Path) else v for k, v in asdict(inputs).items()}
        output = self.graph.invoke({**values, "session_directory": str(session_directory)}, config)
        return GuideResult.from_dict(output["result"]), Path(output["output_directory"])
