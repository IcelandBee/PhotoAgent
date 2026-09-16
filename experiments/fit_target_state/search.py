import numpy as np
from .geometry import PARAMETER_NAMES, decode, geometry_features


def search(renderer,evaluator,reference_size,canvas_size,source_bbox,gt_bbox,config,progress=print):
    rng = np.random.default_rng(config.seed)
    guess = config.initial_guess
    seed = np.array([0.5,0.5,0.0,0.5,0.9,0.4])
    if gt_bbox is not None:
        seed[3:] = geometry_features(gt_bbox)
    initial = seed.copy()
    initial[:3] = guess.viewport_center_x,guess.viewport_center_y,np.log(guess.zoom_scale)
    for i,value in enumerate((guess.subject_bottom_center_x,guess.subject_bottom_y,guess.subject_height),3):
        if value is not None:
            initial[i] = value
    history,rounds = [],[]
    best = None

    def evaluate(parameters,stage):
        nonlocal best
        p,target = decode(parameters,reference_size,canvas_size,source_bbox,config)
        record = {"iteration":len(history),"stage":stage,"parameters":dict(zip(PARAMETER_NAMES,p.tolist())),
                  "zoom_scale":float(np.exp(p[2]))}
        try:
            image,actual,exclusion = renderer.render(target)
            losses,_ = evaluator.evaluate(image,actual,exclusion)
            record["loss"] = losses
            if best is None or losses["total"] < best["loss"]["total"]:
                best = {"parameters":p,"target":target,"loss":losses,"iteration":len(history)}
        except ValueError as exc:
            record["error"] = str(exc)
        history.append(record)

    evaluate(seed,"identity_seed")
    evaluate(initial,"initial_guess")
    dimensions = 6 if config.subject_mode == "reposition" else 3
    # Keep useful identity/GT-supervised basins alive before a global random candidate
    # can win on a repetitive background and pull all refinement into a wrong basin.
    warmup_steps = np.array([0.10,0.10,0.20,0.035,0.035,0.05])
    for anchor in (seed,initial):
        for dimension in range(dimensions):
            for direction in (-1,1):
                p = anchor.copy()
                p[dimension] += direction*warmup_steps[dimension]
                evaluate(p,"seed_neighborhood")
    for _ in range(config.population_size):
        p = seed.copy()
        p[:2] = rng.uniform(*config.center_bounds,size=2)
        p[2] = rng.uniform(*np.log(config.zoom_bounds))
        if dimensions == 6 and rng.random()<0.5:
            p[3:] += rng.normal(0,[0.06,0.06,0.08])
        evaluate(p,"coarse")
    if best is None:
        raise ValueError("Every initial candidate failed rendering; check bounds and image/mask geometry")
    for round_index in range(config.num_rounds):
        steps = np.array([0.12,0.12,0.20,0.035,0.035,0.05])*(0.5**round_index)
        # Local coordinate descent provides a stable improvement even with small populations.
        for dimension in range(dimensions):
            anchor = best["parameters"].copy()
            for direction in (-1,1):
                p = anchor.copy()
                p[dimension] += direction*steps[dimension]
                evaluate(p,f"round_{round_index}_coordinate")
        for _ in range(config.samples_per_round):
            p = best["parameters"].copy()
            p[:dimensions] += rng.normal(0,steps[:dimensions])
            evaluate(p,f"round_{round_index}_random")
        rounds.append({"round":round_index,"evaluations":len(history),"best_iteration":best["iteration"],
                       "best_loss":best["loss"]["total"]})
        progress(f"Round {round_index+1}/{config.num_rounds}: best loss {best['loss']['total']:.6f}")
    if best["loss"]["insufficient_background_penalty"]:
        raise ValueError("No candidate leaves enough valid background; revise ignore masks or bounds")
    return best,history,rounds
