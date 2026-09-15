from pathlib import Path


def detect_reference_subject(state, detector, config):
    observation = detector.detect(state["reference_frame"], Path(config.work_dir) / "subject_mask.png")
    return {"reference_subject": observation}
