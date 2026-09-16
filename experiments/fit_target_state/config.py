from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from photography_viewpoint_agent.schemas.base import NormalizedBox

PositiveInt = Annotated[int, Field(gt=0, strict=True)]
Nonnegative = Annotated[float, Field(ge=0)]


class SettingsModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class LossWeights(SettingsModel):
    subject: Nonnegative = 0.6
    background: Nonnegative = 0.4
    global_: Nonnegative = Field(default=0.0, alias="global")

    @model_validator(mode="after")
    def nonzero(self):
        if self.subject+self.background+self.global_ <= 0:
            raise ValueError("At least one loss weight must be positive")
        return self


class InitialGuess(SettingsModel):
    zoom_scale: float = Field(default=1.0, gt=0)
    viewport_center_x: float = 0.5
    viewport_center_y: float = 0.5
    subject_bottom_center_x: float | None = Field(default=None, ge=0, le=1)
    subject_bottom_y: float | None = Field(default=None, gt=0, le=1)
    subject_height: float | None = Field(default=None, gt=0, le=1)


class FitConfig(SettingsModel):
    subject_mode: Literal["follow_reference", "reposition"] = "reposition"
    device: str = "cpu"
    yolo_model: str = "yolo11n-seg.pt"
    person_confidence: float = Field(default=0.4, gt=0, le=1)
    detection_image_size: int = Field(default=640, ge=320)
    use_manual_gt_bbox: bool = False
    population_size: PositiveInt = 200
    num_rounds: PositiveInt = 4
    samples_per_round: PositiveInt = 200
    seed: int = 42
    preview_max_side: int = Field(default=320, ge=32)
    output_width: PositiveInt | None = None
    output_height: PositiveInt | None = None
    weights: LossWeights = Field(default_factory=LossWeights)
    ignore_rects: list[NormalizedBox] = Field(default_factory=list)
    gt_subject_bbox: NormalizedBox | None = None
    initial_guess: InitialGuess = Field(default_factory=InitialGuess)
    zoom_bounds: tuple[float, float] = (0.4, 3.0)
    center_bounds: tuple[float, float] = (-0.25, 1.25)
    min_background_fraction: float = Field(default=0.15, gt=0, le=1)
    background_color_weight: float = Field(default=0.0, ge=0, le=1)

    @model_validator(mode="after")
    def consistent(self):
        if self.use_manual_gt_bbox and self.gt_subject_bbox is None:
            raise ValueError("use_manual_gt_bbox requires gt_subject_bbox")
        if (self.output_width is None) != (self.output_height is None):
            raise ValueError("Specify both output_width and output_height")
        if not 0 < self.zoom_bounds[0] < self.zoom_bounds[1]:
            raise ValueError("zoom_bounds must be positive and increasing")
        if not self.center_bounds[0] < self.center_bounds[1]:
            raise ValueError("center_bounds must be increasing")
        if not self.zoom_bounds[0] <= self.initial_guess.zoom_scale <= self.zoom_bounds[1]:
            raise ValueError("initial_guess.zoom_scale must be inside zoom_bounds")
        if self.subject_mode == "follow_reference" and self.weights.background+self.weights.global_ <= 0:
            raise ValueError("follow_reference needs a background or global loss weight")
        return self
