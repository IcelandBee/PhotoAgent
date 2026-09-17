import math
from PIL import Image
from photography_viewpoint_agent.config.settings import AgentConfig
from photography_viewpoint_agent.schemas.target import TargetState
from photography_viewpoint_agent.schemas.rendering import RenderMeta
from photography_viewpoint_agent.schemas.validation import SketchValidation
from photography_viewpoint_agent.renderer.viewpoint_warp import rotation_homography, warp_bbox
from photography_viewpoint_agent.tools.geometry import (
    fit_viewport, contain_subject, transform_bbox_by_viewport, clip_bbox_to_canvas,
    expected_subject_bbox, subject_is_noop,
)


def boxes_match(left, right, tolerance=1e-8):
    if left is None or right is None:
        return left is None and right is None
    return all(abs(a-b) <= tolerance for a,b in zip(left,right))


def validate_sketch(target: TargetState, meta: RenderMeta, path: str,
                    config: AgentConfig) -> SketchValidation:
    messages = []
    expected_viewport = target.framing.reference_viewport
    if meta.reference_size is not None:
        expected_viewport, _ = fit_viewport(expected_viewport, meta.reference_size,
                                             (config.target_width, config.target_height))
        if any(abs(a-b) > 1e-8 for a,b in zip(expected_viewport, target.framing.reference_viewport)):
            messages.append("Viewport was aspect-fitted around its center; framing error uses the fitted viewport.")
    framing = sum(abs(a - b) for a, b in zip(expected_viewport, meta.rendered_viewport)) / 4
    passed = framing <= config.framing_threshold
    viewpoint = target.viewpoint
    if viewpoint.mode == 'depth_3d' or viewpoint.active:
        if meta.viewpoint_warp is None or meta.viewpoint_warp.get('parameters') != viewpoint.model_dump():
            passed = False
            messages.append('Viewpoint metadata does not match requested transform.')
        if viewpoint.mode == 'depth_3d':
            warp = meta.viewpoint_warp or {}
            if warp.get('backend') != 'depth_3d' or not 0 < warp.get('valid_fraction',0) <= 1:
                passed = False
                messages.append('Missing depth_3d coverage/backend metadata.')
            messages.append('Depth coverage is diagnostic; PASS validates geometry plumbing, not novel-view realism.')
    if not passed:
        messages.append("Framing error exceeds threshold.")
    position = scale = None
    if meta.subject_mode != target.subject.mode:
        passed = False
        messages.append("Rendered subject mode does not match target mode.")
    if meta.requested_viewport is not None and not boxes_match(meta.requested_viewport, target.framing.reference_viewport):
        passed = False
        messages.append("Requested viewport metadata does not match target.")
    if meta.source_subject_bbox is not None:
        source_box = meta.source_subject_bbox
        if viewpoint.mode == 'depth_3d':
            source_box = (meta.viewpoint_warp or {}).get('projected_subject_bbox')
            if target.subject.mode == 'reposition':
                messages.append('Natural subject projection omitted: background depth repaired; subject rendered independently.')
        elif viewpoint.active and meta.reference_size is not None:
            h,_,_ = rotation_homography(meta.reference_size,viewpoint)
            source_box = warp_bbox(source_box,meta.reference_size,h)
        projected = transform_bbox_by_viewport(source_box, meta.rendered_viewport) if source_box else None
        if not boxes_match(projected, meta.natural_subject_bbox):
            passed = False
            messages.append("Natural subject bbox does not match source-to-viewport projection.")
    if target.subject.mode == "follow_reference":
        if meta.subject_transform_applied:
            passed = False
            messages.append("follow_reference must not apply an independent subject transform.")
        if meta.source_subject_bbox is None:
            messages.append("Subject detection skipped; only framing and image integrity are validated.")
            if meta.natural_subject_bbox is not None or meta.rendered_subject_bbox is not None:
                passed = False
                messages.append("Subject bbox metadata lacks a source observation.")
        elif not boxes_match(clip_bbox_to_canvas(meta.natural_subject_bbox), meta.rendered_subject_bbox):
            passed = False
            messages.append("Follow-reference subject does not match its natural visible bbox.")
        else:
            messages.append("Subject follows the reference viewport without independent editing.")
    elif meta.rendered_subject_bbox is None:
        passed = False
        messages.append("Subject rendering is missing.")
    else:
        wanted, actual = target.subject.bbox, meta.rendered_subject_bbox
        position = math.hypot((wanted[0] + wanted[2] - actual[0] - actual[2]) / 2,
                              wanted[3] - actual[3])
        expected_height = wanted[3] - wanted[1]
        if meta.source_subject_bbox is not None and meta.reference_size is not None:
            source = meta.source_subject_bbox
            rw, rh = meta.reference_size
            sw, sh = (source[2]-source[0])*rw, (source[3]-source[1])*rh
            factor, _, _ = contain_subject((sw, sh), wanted, (config.target_width, config.target_height))
            if not meta.subject_transform_applied:
                expected = expected_subject_bbox(source, meta.reference_size, wanted,
                                                  (config.target_width, config.target_height))
                if not subject_is_noop(meta.natural_subject_bbox, expected,
                                      config.subject_noop_position_threshold, config.subject_noop_scale_threshold):
                    passed = False
                    messages.append("Subject transform was skipped outside the no-op thresholds.")
                if not boxes_match(clip_bbox_to_canvas(meta.natural_subject_bbox), actual):
                    passed = False
                    messages.append("No-op rendered bbox does not match natural geometry.")
                messages.append("Independent subject editing skipped by no-op fast path.")
            expected_height = sh*factor/config.target_height
            if expected_height < wanted[3]-wanted[1]-1e-8:
                messages.append("Subject height is limited by contain scaling within the target envelope.")
            # Compare width in pixels with ratio-preserving width; permit raster rounding.
            ratio_error_px = abs((actual[2]-actual[0])*config.target_width -
                                 (actual[3]-actual[1])*config.target_height*sw/sh)
            if ratio_error_px > 2 + 2*sw/sh:
                passed = False
                messages.append("Subject aspect ratio is distorted.")
        else:
            passed = False
            messages.append("Source subject geometry is missing; contain scale cannot be verified.")
        scale = abs(expected_height - (actual[3] - actual[1]))
        tolerance_x, tolerance_y = 1/config.target_width, 1/config.target_height
        if (actual[0] < wanted[0]-tolerance_x or actual[2] > wanted[2]+tolerance_x or
                actual[1] < wanted[1]-tolerance_y or actual[3] > wanted[3]+tolerance_y):
            passed = False
            messages.append("Rendered subject exceeds target envelope.")
        if position > config.subject_position_threshold:
            passed = False
            messages.append("Subject position error exceeds threshold.")
        if scale > config.subject_scale_threshold:
            passed = False
            messages.append("Subject scale error exceeds threshold.")
    try:
        with Image.open(path) as image:
            image.load()
            if image.size != (config.target_width, config.target_height):
                passed = False
                messages.append("Target sketch dimensions do not match configuration.")
    except (OSError, ValueError) as exc:
        passed = False
        messages.append(f"Target sketch cannot be read: {exc}")
    return SketchValidation(passed=passed, subject_position_error=position,
                            subject_scale_error=scale, framing_error=framing, messages=messages)
