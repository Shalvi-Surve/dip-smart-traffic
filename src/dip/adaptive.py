"""Modular, deterministic rule-based analysis, planning and execution."""
import math
import re
import time
import numpy as np
from PIL import Image
from analysis.quality import calculate_quality, estimate_noise
from analysis.degradation import assess_degradation
from .dehaze import dark_channel_prior
from .frequency import gaussian_lowpass_luminance
from .spatial import (bilateral_denoise, clahe_luminance, edge_enhance, edge_map,
                      equalize_luminance, gamma_correct, gaussian_denoise,
                      median_denoise, morphology, normalize_luminance,
                      unsharp_luminance)


def _gamma_for_mean(mean, target):
    """Choose gamma by mapping the measured mean toward a conservative target."""
    x = min(254.999, max(0.001, float(mean))) / 255.0
    y = float(target) / 255.0
    return float(np.clip(math.log(y) / math.log(x), 0.55, 1.60))


def _algorithm_id(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def normalize_pipeline_result(result):
    """Return a validated pipeline result using the canonical stage schema.

    Input is the mapping returned by :func:`build_pipeline` (or a compatible
    cached mapping). Required result keys are ``original``, ``final``,
    ``input_metrics``, ``plan``, and ``stages``. Each stage requires a name,
    real input/output PIL images, and before/after metric mappings. The
    canonical output stage keys are ``index``, ``name``, ``algorithm_id``,
    ``input_image``, ``image``/``output_image``, ``parameters``, ``reason``,
    ``before``, ``after``, ``processing_ms``, and
    ``changed_pixel_fraction``. Older records can resolve a missing ID only
    from the planned action with the same operation name; no identifier is
    fabricated. Pixel-change fields are measured from the saved arrays when
    absent. Invalid required data raises ``ValueError`` with stage context.
    """
    if not isinstance(result, dict):
        raise ValueError("Invalid pipeline result: expected a result mapping")
    missing_result_keys = [key for key in ("original", "final", "input_metrics", "plan", "stages")
                           if key not in result]
    if missing_result_keys:
        raise ValueError("Invalid pipeline result: missing required field(s): " + ", ".join(missing_result_keys))
    if not isinstance(result["stages"], list):
        raise ValueError("Invalid pipeline result: 'stages' must be a list")
    for field in ("original", "final"):
        if not isinstance(result[field], Image.Image):
            raise ValueError(f"Invalid pipeline result: '{field}' must be a PIL image")
    if not isinstance(result["input_metrics"], dict):
        raise ValueError("Invalid pipeline result: 'input_metrics' must be a metrics mapping")
    plan = result["plan"]
    if not isinstance(plan, dict) or not isinstance(plan.get("actions"), list):
        raise ValueError("Invalid pipeline result: 'plan.actions' must be a list")
    actions = plan["actions"]
    actions_by_name = {a.get("operation"): a for a in actions if isinstance(a, dict)}
    normalized = []
    for position, source in enumerate(result["stages"], start=1):
        if not isinstance(source, dict):
            raise ValueError(f"Invalid pipeline stage {position}: expected a stage record")
        stage = dict(source)
        name = stage.get("name") or stage.get("stage_name")
        action = actions_by_name.get(name, {})
        algorithm_id = stage.get("algorithm_id") or action.get("algorithm_id")
        if not name:
            raise ValueError(f"Invalid pipeline stage {position}: required stage name is missing")
        if not algorithm_id:
            raise ValueError(f"Invalid pipeline stage {position} ({name}): required algorithm_id is missing from stage and plan")
        output = stage.get("image")
        if output is None:
            output = stage.get("output_image")
        stage_input = stage.get("input_image")
        if output is None or stage_input is None:
            missing = "output image" if output is None else "input image"
            raise ValueError(f"Invalid pipeline stage {position} ({name}): required {missing} is missing")
        if not isinstance(output, Image.Image) or not isinstance(stage_input, Image.Image):
            raise ValueError(f"Invalid pipeline stage {position} ({name}): input/output must be PIL images")
        if output.size != stage_input.size:
            raise ValueError(f"Invalid pipeline stage {position} ({name}): input/output dimensions differ")
        stage["index"] = int(stage.get("index", stage.get("stage_index", position)))
        stage["stage_index"] = int(stage.get("stage_index", stage["index"]))
        stage["name"] = name
        stage["stage_name"] = stage.get("stage_name", name)
        stage["algorithm_id"] = algorithm_id
        stage["image"] = output
        stage["output_image"] = output
        stage["input_image"] = stage_input
        stage["parameters"] = stage.get("parameters", action.get("parameters", {}))
        stage["reason"] = stage.get("reason", action.get("reason", ""))
        stage["explanation"] = stage.get("explanation", action.get("explanation", ""))
        stage["limitation"] = stage.get("limitation", action.get("limitation", ""))
        stage["before"] = stage.get("before")
        stage["after"] = stage.get("after")
        stage["output_kind"] = stage.get("output_kind", "photograph")
        duration = stage.get("processing_ms", stage.get("duration_ms"))
        stage["processing_ms"] = duration
        stage["duration_ms"] = stage.get("duration_ms", duration)
        stage["diagnostics"] = stage.get("diagnostics", {})
        if stage.get("changed_pixel_fraction") is None:
            a = np.asarray(stage_input.convert("RGB"), dtype=np.int16)
            b = np.asarray(output.convert("RGB"), dtype=np.int16)
            stage["changed_pixel_fraction"] = float((np.max(np.abs(a - b), axis=2) > 0).mean())
        if stage.get("mean_absolute_rgb_change") is None:
            a = np.asarray(stage_input.convert("RGB"), dtype=np.int16)
            b = np.asarray(output.convert("RGB"), dtype=np.int16)
            stage["mean_absolute_rgb_change"] = float(np.abs(a - b).mean())
        if stage.get("changed_pixel_count") is None:
            a = np.asarray(stage_input.convert("RGB"), dtype=np.int16)
            b = np.asarray(output.convert("RGB"), dtype=np.int16)
            stage["changed_pixel_count"] = int((np.max(np.abs(a - b), axis=2) > 0).sum())
        if not isinstance(stage["before"], dict) or not isinstance(stage["after"], dict):
            raise ValueError(f"Invalid pipeline stage {position} ({name}): before/after measurements are missing")
        normalized.append(stage)
    result = dict(result)
    result["stages"] = normalized
    return result


def analyze_and_plan(image, denoise_choice="automatic", dehaze_choice="automatic", source_metadata=None,
                     edge_diagnostic=False, morphology_choice="off", selected_operations=None):
    """Analyze once and produce an inspectable rule-based processing plan.

    The haze-like indicator is a weighted dark-channel/contrast/saturation
    heuristic, not a probability or a trained weather classifier.
    """
    if selected_operations and denoise_choice in {"automatic", "off"} and "Denoising" in selected_operations:
        # A fixed operation list requires a concrete denoiser. Bilateral is the
        # conservative default; the UI still lets the user choose another one.
        denoise_choice = "bilateral"
    if denoise_choice not in {"automatic", "median", "gaussian", "bilateral", "frequency", "off"}:
        raise ValueError("Unknown denoising choice")
    if dehaze_choice not in {"automatic", "force", "off"}:
        raise ValueError("Unknown dehazing choice")
    if morphology_choice not in {"off", "erosion", "dilation", "opening", "closing"}:
        raise ValueError("Unknown morphology choice")
    source_mode = image.info.get("source_mode", image.mode)
    original = image.convert("RGB").copy()
    original.info["source_mode"] = source_mode
    assessment = assess_degradation(original)
    m = assessment["metrics"]
    noise = assessment["noise_estimate"]
    actions = []

    def action(op, selected, reason, explanation, parameters=None, limitation=""):
        actions.append({"operation": op, "algorithm_id": _algorithm_id(op),
                        "status": "planned" if selected else "skipped",
                        "selected": bool(selected), "reason": reason, "explanation": explanation,
                        "parameters": parameters or {}, "limitation": limitation})

    robust_range = m["robust_dynamic_range"]
    normalize = robust_range < 225
    action("Robust luminance normalization", normalize,
           (f"The 1st–99th percentile luminance span is {robust_range:.1f}/255; stretching this compressed range may recover usable tonal separation."
            if normalize else f"The 1st–99th percentile luminance span is already {robust_range:.1f}/255; a stretch risks unnecessary clipping."),
           "Linearly map the 1st–99th luminance percentiles to the display range while preserving YCbCr chroma.",
           {"low_percentile": 1.0, "high_percentile": 99.0},
           "Can clip scene details outside the selected percentile interval and is not a visibility score.")

    haze_gate = (m["haze_like_indicator"] >= 0.58 and m["mean_brightness"] >= 75
                 and m["edge_fraction_gradient_ge_4"] >= 0.005
                 and m["dark_channel_mean"] >= 0.22)
    haze_selected = dehaze_choice == "force" or (dehaze_choice == "automatic" and haze_gate)
    if dehaze_choice == "force":
        haze_reason = "Explicitly selected by the user; dehazing is experimental for this input."
    elif dehaze_choice == "off":
        haze_reason = "Disabled by the user."
    elif haze_selected:
        haze_reason = (f"Haze-like indicator {m['haze_like_indicator']:.3f}, dark-channel mean "
                       f"{m['dark_channel_mean']:.3f}, and edge fraction >=4 {m['edge_fraction_gradient_ge_4']:.3f} meet the conservative heuristic.")
    else:
        haze_reason = (f"Heuristic gate not met (indicator {m['haze_like_indicator']:.3f}; requires >= 0.58, brightness >= 75, dark channel >= 0.22, edge fraction >=4 >= 0.005).")
    action("Dark-channel-prior dehazing", haze_selected,
           haze_reason,
           "Estimate atmospheric light and transmission using the dark-channel prior, invert the scattering model, then blend a conservative portion of the recovered radiance with the input.",
           {"window": 15, "omega": 0.70, "transmission_floor": 0.55, "strength": 0.65},
           "May over-correct skies, bright objects, night scenes, or non-hazy images; the indicator is not definitive.")

    mean = m["mean_brightness"]
    if mean < 75:
        if m["clipped_shadow_fraction"] >= 0.98:
            action("Gamma correction", False,
                   f"Mean luminance {mean:.1f} is low, but {m['clipped_shadow_fraction']:.1%} of pixels are clipped at black; gamma cannot recover absent detail.",
                   "No transform is applied to an almost entirely clipped-black image.",
                   limitation="Information lost to clipping cannot be reconstructed.")
        else:
            gamma = _gamma_for_mean(mean, 105)
            action("Gamma correction", True,
                   f"Measured mean luminance {mean:.1f}/255 is below 75; gamma maps the mean toward 105/255.",
                   "Apply a power-law transform to Y in YCbCr while retaining chroma.", {"gamma": gamma, "target_mean": 105},
                   "Gamma can lift noise and clipped shadows cannot be recovered.")
    elif mean > 185:
        if m["clipped_highlight_fraction"] >= 0.98:
            action("Gamma correction", False,
                   f"Mean luminance {mean:.1f} is high, but {m['clipped_highlight_fraction']:.1%} of pixels are clipped at white; gamma cannot recover highlight detail.",
                   "No transform is applied to an almost entirely clipped-white image.",
                   limitation="Information lost to clipping cannot be reconstructed.")
        else:
            gamma = _gamma_for_mean(mean, 165)
            action("Gamma correction", True,
                   f"Measured mean luminance {mean:.1f}/255 is above 185; gamma maps the mean toward 165/255.",
                   "Apply a power-law transform to Y in YCbCr while retaining chroma.", {"gamma": gamma, "target_mean": 165},
                   "Gamma cannot recover clipped highlights and can darken useful bright details.")
    else:
        action("Gamma correction", False,
               f"Mean luminance {mean:.1f}/255 is in the 75–185 operating interval.",
               "No global illumination transform is needed by the current rule.", limitation="Global thresholds do not describe local exposure.")

    contrast = m["contrast_stddev"]
    global_eq_selected = False
    local_low = m["local_contrast_stddev"] < 20
    clahe_selected = (contrast < 42 or local_low) and robust_range > 1 and not haze_selected
    if haze_selected:
        clahe_reason = "Skipped to avoid stacking CLAHE on selected dehazing contrast recovery."
    elif robust_range <= 1:
        clahe_reason = "Skipped because the image has effectively no robust luminance range; equalization would be a no-op and cannot recover missing detail."
    elif contrast < 42 or local_low:
        clahe_reason = (f"Low contrast evidence (global σ {contrast:.1f} below 42 or local tile σ "
                        f"{m['local_contrast_stddev']:.1f} below 20); CLAHE is selected for local recovery.")
    else:
        clahe_reason = f"Global luminance standard deviation {contrast:.1f} is at least 42."
    action("Global histogram equalization", global_eq_selected,
           "Skipped by the adaptive policy; low-contrast inputs use CLAHE to limit global tone changes, while dehazed inputs avoid a second equalizer.",
           "Equalize the global luminance histogram while retaining YCbCr chroma.",
           {}, "Can over-expand noise and does not guarantee improved visibility.")
    action("CLAHE on luminance", clahe_selected, clahe_reason,
           "Clip and redistribute local luminance histograms with bilinear tile interpolation.",
           {"tile_size": 64, "clip_limit": 2.0},
           "Can amplify noise or create local halos; thresholds are fixed demonstrator rules.")
    if denoise_choice == "median":
        denoise_op, denoise_params, denoise_reason = "Median denoising", {"kernel_size": 3, "selection": "user-selected"}, "Explicitly selected by the user."
        denoise_selected = True
    elif denoise_choice == "gaussian":
        denoise_op, denoise_params, denoise_reason = "Gaussian denoising", {"radius": 1.0, "selection": "user-selected"}, "Explicitly selected by the user."
        denoise_selected = True
    elif denoise_choice == "bilateral":
        denoise_op, denoise_params, denoise_reason = "Bilateral denoising", {"diameter": 5, "sigma_color": 25.0, "sigma_space": 2.0, "selection": "user-selected"}, "Explicitly selected by the user."
        denoise_selected = True
    elif denoise_choice == "frequency":
        denoise_op, denoise_params, denoise_reason = "FFT Gaussian low-pass", {"sigma": 0.8, "selection": "user-selected"}, "Explicitly selected by the user."
        denoise_selected = True
    elif denoise_choice == "automatic" and noise >= 8:
        sigma = float(np.clip(0.5 + (noise - 8.0) / 25.0, 0.5, 1.2))
        denoise_op, denoise_params, denoise_reason = "FFT Gaussian low-pass", {"sigma": sigma, "noise_estimate": noise, "rule": "noise estimate >= 8"}, f"Median-residual MAD proxy {noise:.2f} is >= 8; a mild Gaussian frequency filter is selected."
        denoise_selected = True
    else:
        denoise_op, denoise_params = ("Denoising", {"choice": denoise_choice, "noise_estimate": noise})
        denoise_selected = False
        denoise_reason = (f"Noise proxy {noise:.2f} is below 8; automatic denoising skipped."
                          if denoise_choice == "automatic" else "Explicitly skipped by the user.")
    action(denoise_op, denoise_selected, denoise_reason,
           ({"Median denoising": "Remove isolated impulse noise with a 3x3 median filter on luminance while retaining chroma.",
             "Gaussian denoising": "Smooth fine-scale approximately Gaussian noise on luminance.",
             "Bilateral denoising": "Combine spatial and luminance-range weights to suppress noise while retaining stronger edges.",
             "FFT Gaussian low-pass": "Attenuate high-frequency luminance components using a Gaussian Fourier transfer function and inverse FFT."}.get(denoise_op, "No spatial/frequency denoising filter is applied.")),
           denoise_params, "Fine texture and narrow edges can be smoothed; the noise estimate also responds to texture.")

    sharp = m["sharpness_laplacian_variance"]
    has_edges = m["edge_fraction_gradient_ge_4"] >= 0.005
    sharp_selected = sharp < 180 and has_edges and (noise < 8 or denoise_selected)
    if sharp >= 180:
        sharp_reason = f"Input Laplacian variance {sharp:.1f} is at least 180."
    elif noise >= 8 and not denoise_selected:
        sharp_reason = "Skipped because the noise proxy is high and denoising was not selected; sharpening could amplify noise."
    elif not has_edges:
        sharp_reason = "Skipped because the image has too little measured edge activity to sharpen meaningfully."
    else:
        sharp_reason = f"Input Laplacian variance {sharp:.1f} is below 180 with no unresolved high-noise gate."
    action("Luminance unsharp mask", sharp_selected, sharp_reason,
           "Apply one thresholded unsharp mask to luminance after restoration/denoising.",
           {"radius": 1.2, "percent": 90, "threshold": 5},
           "Can create halos or emphasize compression artifacts; sharpness is not perceptual quality.")
    edge_boost = 180 <= sharp < 450 and has_edges and noise < 8 and m["local_contrast_stddev"] < 20
    action("Sobel edge enhancement", edge_boost,
           (f"Laplacian variance {sharp:.1f} is moderate, local contrast is {m['local_contrast_stddev']:.1f}, and the noise proxy is modest; a bounded gradient lift is selected instead of unsharp masking."
            if edge_boost else "Skipped to avoid redundant sharpening, noise amplification, or boosting insufficient edge evidence."),
           "Add a bounded Sobel gradient-magnitude term to luminance while retaining chroma.",
           {"strength": 0.25, "operator": "sobel"},
           "Can accentuate noise and halos; this is a mild edge emphasis, not visibility validation.")
    action("Rain-streak suppression", False,
           "No rain-specific streak detector/suppression algorithm is implemented; generic smoothing is not labeled rain removal.",
           "Reserved as a future, separately evaluated method.", limitation="Unavailable in this build.")
    actions[-1]["status"] = "unavailable"
    if edge_diagnostic:
        action("Sobel edge diagnostic", True,
               "Requested as an auxiliary analysis image; it is not substituted for the natural-color output.",
               "Compute Sobel horizontal/vertical gradients and return their magnitude as an L-mode diagnostic.",
               {"operator": "sobel"}, "Gradient maps emphasize edges and noise; they are not enhanced traffic photographs.")
    if morphology_choice != "off":
        action("Morphological diagnostic", True,
               f"The user requested the {morphology_choice} operation as an auxiliary grayscale diagnostic.",
               "Apply the selected grayscale morphology operation to luminance; it remains separate from the final RGB photograph.",
               {"operation": morphology_choice, "size": 3}, "Morphological masks alter structures and are not natural-color enhancement outputs.")
    else:
        action("Morphological diagnostic", False,
               "Skipped by default because morphology is a structural diagnostic and is not suitable as a final traffic photograph.",
               "Optional grayscale erosion, dilation, opening, or closing for controlled experiments.",
               {"choice": "off"}, "Auxiliary output only.")
    if selected_operations is not None:
        selectable = {
            "Robust luminance normalization", "Dark-channel-prior dehazing", "Gamma correction",
            "Global histogram equalization", "CLAHE on luminance", "Denoising",
            "Median denoising", "Gaussian denoising", "Bilateral denoising", "FFT Gaussian low-pass",
            "Luminance unsharp mask", "Sobel edge enhancement",
        }
        requested = list(selected_operations)
        if len(set(requested)) != len(requested) or any(name not in selectable for name in requested):
            raise ValueError("Fixed pipeline contains a duplicate or unsupported operation")
        # 'Denoising' is a user-facing family choice. Resolve it to the concrete
        # implementation selected above (or the adaptive concrete action).
        actions_by_operation = {item["operation"]: item for item in actions}
        denoise_action = next((item for item in actions if item["operation"] in {
            "Denoising", "Median denoising", "Gaussian denoising", "Bilateral denoising", "FFT Gaussian low-pass"
        }), None)
        resolved = []
        for name in requested:
            action_name = denoise_action["operation"] if name == "Denoising" and denoise_action else name
            if action_name not in actions_by_operation:
                raise ValueError(f"Fixed pipeline operation is unavailable: {name}")
            item = actions_by_operation[action_name]
            was_selected = item["selected"]
            item["selected"] = True
            item["status"] = "planned"
            item["parameters"] = dict(item.get("parameters") or {})
            if name == "Gamma correction" and "gamma" not in item["parameters"]:
                if mean < 125:
                    target_mean = 105
                    gamma = _gamma_for_mean(mean, target_mean)
                elif mean > 155:
                    target_mean = 145
                    gamma = _gamma_for_mean(mean, target_mean)
                else:
                    target_mean, gamma = mean, 1.0
                item["parameters"].update({"gamma": gamma, "target_mean": target_mean})
                item["reason"] = (f"Explicitly selected in the fixed sequence; measured mean luminance is {mean:.1f}/255, "
                                   f"with a conservative target of {target_mean:.1f}/255.")
            if name == "Denoising":
                item["parameters"]["selection"] = "user-selected"
                item["reason"] = "Explicitly selected in the fixed operation sequence; " + item["reason"]
            elif name == "Dark-channel-prior dehazing" and not was_selected:
                item["parameters"]["selection"] = "user-selected"
                item["reason"] = "Explicitly selected in the fixed operation sequence; experimental dehazing is forced."
            elif name == "Dark-channel-prior dehazing":
                item["parameters"]["selection"] = "user-selected"
                item["reason"] = "Explicitly selected in the fixed operation sequence; " + item["reason"]
            else:
                item["reason"] = "Explicitly selected in the fixed operation sequence; " + item["reason"]
            resolved.append(action_name)
        for item in actions:
            if (item["operation"] not in resolved and item["selected"]
                    and item["operation"] not in {"Sobel edge diagnostic", "Morphological diagnostic"}):
                item["selected"] = False
                item["status"] = "skipped"
                item["reason"] = "Not selected in the user-configured fixed operation sequence. " + item["reason"]
        selected_actions = [actions_by_operation[name] for name in resolved]
        other_actions = [item for item in actions if item["operation"] not in resolved]
        actions = selected_actions + other_actions
    selected_order = [item["operation"] for item in actions if item["selected"]]
    effective_dehaze_choice = ("force" if selected_operations is not None
                               and "Dark-channel-prior dehazing" in selected_operations
                               else dehaze_choice)
    return {"assessment": assessment, "actions": actions, "source_metadata": source_metadata or {},
            "denoise_choice": denoise_choice, "dehaze_choice": effective_dehaze_choice,
            "selection_mode": "fixed-order" if selected_operations is not None else "adaptive",
            "selected_operation_order": selected_order,
            "engine": "deterministic rule-based adaptive DIP"}


def build_pipeline(image, denoise_choice="automatic", dehaze_choice="automatic", source_metadata=None,
                   edge_diagnostic=False, morphology_choice="off", selected_operations=None,
                   progress_callback=None):
    """Plan then execute real stages; optionally report synchronous progress events.

    The callback receives ``plan_ready``, ``stage_started``, ``stage_completed``,
    ``stage_skipped``, ``stage_failed`` and ``complete`` mappings. Events carry
    the current image and real stage record where applicable. Callbacks are
    presentation hooks only and cannot alter pipeline arrays.
    """
    started = time.perf_counter()
    source_mode = image.info.get("source_mode", image.mode)
    original = image.convert("RGB").copy()
    original.info["source_mode"] = source_mode
    plan = analyze_and_plan(original, denoise_choice=denoise_choice, dehaze_choice=dehaze_choice,
                            source_metadata=source_metadata, edge_diagnostic=edge_diagnostic,
                            morphology_choice=morphology_choice, selected_operations=selected_operations)
    current = original.copy()
    walkthrough_current = original.copy()
    current_metrics = plan["assessment"]["metrics"]
    walkthrough_metrics = current_metrics
    stages = []
    selected_actions = [a for a in plan["actions"] if a["selected"]]
    if progress_callback:
        progress_callback({"event": "plan_ready", "total": len(selected_actions),
                           "original": original.copy(), "plan": plan})
    for operation_number, action in enumerate(selected_actions, start=1):
        if not action["selected"]:
            continue
        op_start = time.perf_counter()
        action["status"] = "processing"
        if progress_callback:
            progress_callback({"event": "stage_started", "operation_number": operation_number,
                               "total": len(selected_actions), "operation": action["operation"],
                               "current_image": walkthrough_current.copy(), "original": original.copy()})
        try:
            name = action["operation"]
            params = action["parameters"]
            diagnostic = name in {"Sobel edge diagnostic", "Morphological diagnostic"}
            stage_input = walkthrough_current if diagnostic else current
            before = walkthrough_metrics if diagnostic else current_metrics
            if name == "Robust luminance normalization":
                result = normalize_luminance(current, **params)
            elif name == "Dark-channel-prior dehazing":
                result, diagnostics = dark_channel_prior(current, **params)
                action["diagnostics"] = diagnostics
            elif name == "Gamma correction":
                result = gamma_correct(current, params["gamma"])
            elif name == "CLAHE on luminance":
                result = clahe_luminance(current, tile_size=params["tile_size"], clip_limit=params["clip_limit"])
            elif name == "Global histogram equalization":
                result = equalize_luminance(current)
            elif name == "Median denoising":
                result = median_denoise(current, size=params["kernel_size"])
            elif name == "Gaussian denoising":
                result = gaussian_denoise(current, radius=params["radius"])
            elif name == "Bilateral denoising":
                result = bilateral_denoise(current, **{k: params[k] for k in ("diameter", "sigma_color", "sigma_space")})
            elif name == "FFT Gaussian low-pass":
                result = gaussian_lowpass_luminance(current, sigma=params["sigma"])
            elif name == "Luminance unsharp mask":
                result = unsharp_luminance(current, radius=params["radius"], percent=params["percent"], threshold=params["threshold"])
            elif name == "Sobel edge diagnostic":
                result = edge_map(stage_input, **params)
            elif name == "Morphological diagnostic":
                result = morphology(stage_input, **params)
            elif name == "Sobel edge enhancement":
                result = edge_enhance(current, **params)
            else:
                action["status"] = "failed"
                action["error"] = "No implementation is registered for this planned operation."
                continue
            if not diagnostic:
                result = result.convert("RGB")
            after = calculate_quality(result)
            difference = np.abs(np.asarray(result.convert("RGB"), dtype=np.int16)
                                - np.asarray(stage_input.convert("RGB"), dtype=np.int16))
            delta = float(difference.mean())
            changed_fraction = float((difference.max(axis=2) > 0).mean())
            if changed_fraction == 0:
                action["status"] = "skipped"
                action["selected"] = False
                action["reason"] += " The algorithm was evaluated, but 8-bit quantization produced an identical image; it is not counted as an executed stage."
                action["processing_ms"] = (time.perf_counter() - op_start) * 1000.0
                if progress_callback:
                    progress_callback({"event": "stage_skipped", "operation_number": operation_number,
                                       "total": len(selected_actions), "operation": name,
                                       "reason": action["reason"], "current_image": walkthrough_current.copy(),
                                       "original": original.copy()})
                continue
            elapsed = (time.perf_counter() - op_start) * 1000.0
            action["status"] = "executed"
            action["processing_ms"] = elapsed
            stage_record = {"index": len(stages) + 1, "stage_index": len(stages) + 1,
                           "name": name, "stage_name": name, "algorithm_id": action["algorithm_id"],
                           "explanation": action["explanation"],
                           "reason": action["reason"], "parameters": params, "limitation": action["limitation"],
                           "before": before, "after": after, "image": result,
                           "output_image": result,
                           "input_image": stage_input.copy(), "output_kind": "diagnostic" if diagnostic else "photograph",
                           "mean_absolute_rgb_change": delta, "changed_pixel_fraction": changed_fraction,
                           "changed_pixel_count": int((difference.max(axis=2) > 0).sum()),
                           "processing_ms": elapsed, "duration_ms": elapsed,
                           "diagnostics": action.get("diagnostics", {})}
            stages.append(stage_record)
            walkthrough_current = result
            walkthrough_metrics = after
            if not diagnostic:
                current = result
                current_metrics = after
            if progress_callback:
                progress_callback({"event": "stage_completed", "operation_number": operation_number,
                                   "total": len(selected_actions), "operation": name,
                                   "stage": stage_record, "current_image": walkthrough_current.copy(),
                                   "original": original.copy()})
        except Exception as exc:  # Stage failures are explicit; preserve the last valid image.
            action["status"] = "failed"
            action["error"] = f"{type(exc).__name__}: {exc}"
            action["processing_ms"] = (time.perf_counter() - op_start) * 1000.0
            if progress_callback:
                progress_callback({"event": "stage_failed", "operation_number": operation_number,
                                   "total": len(selected_actions), "operation": action["operation"],
                                   "error": action["error"], "current_image": walkthrough_current.copy(),
                                   "original": original.copy()})
    elapsed = (time.perf_counter() - started) * 1000.0
    skipped = [a["reason"] for a in plan["actions"] if a["status"] in {"skipped", "failed", "unavailable"}]
    pipeline_result = normalize_pipeline_result({"original": original, "stages": stages, "final": current.copy(),
            "decisions": skipped, "input_metrics": plan["assessment"]["metrics"],
            "noise_estimate": plan["assessment"]["noise_estimate"], "analysis": plan["assessment"],
            "plan": plan, "total_processing_ms": elapsed})
    if progress_callback:
        progress_callback({"event": "complete", "total": len(selected_actions),
                           "completed": len(stages), "result": pipeline_result})
    return pipeline_result


def stage_view(result, stage_index):
    """Return an immutable view of original/current/previous image for navigation."""
    result = normalize_pipeline_result(result)
    if stage_index < 0 or stage_index > len(result["stages"]):
        raise IndexError("stage index out of range")
    original = result["original"]
    if stage_index == 0:
        return {"index": 0, "name": "Original", "image": original, "metrics": result["input_metrics"],
                "previous_metrics": None}
    stage = result["stages"][stage_index - 1]
    return {"index": stage_index, "name": stage["name"], "image": stage["image"],
            "metrics": stage["after"], "previous_metrics": stage["before"], "stage": stage}
