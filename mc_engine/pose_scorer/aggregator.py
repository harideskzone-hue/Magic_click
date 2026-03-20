def _effective_confidence(conf: float, config: dict) -> float:
    """
    AGGREGATION_FORMULA confidence adjustment:
      c >= 0.75: full weight
      0.55 <= c < 0.75: c * 0.60 (downweight)
      c < 0.55: skip entirely (return 0)
    """
    low_threshold   = config['CONFIDENCE']['low_threshold']      # 0.75
    min_to_score    = config['CONFIDENCE']['min_to_score']        # 0.55
    low_weight_fac  = config['CONFIDENCE']['low_weight_factor']   # 0.60

    if conf >= low_threshold:
        return conf
    elif conf >= min_to_score:
        return conf * low_weight_fac
    else:
        return 0.0  # skip


def _score_band(score: float, config: dict) -> str:
    for s_min, s_max, label in config['SCORE_BANDS']:
        if s_min <= score <= s_max:
            return label
    return "Unknown"


def _aggregate_group(modules: dict, weights: dict, config: dict) -> float | None:
    """
    LEVEL 1 aggregation — within a group.
    Missing/skipped modules have their weights redistributed.
    """
    num = 0.0
    den = 0.0
    for mod_name, mod_res in modules.items():
        if mod_res.get('skipped', True):
            continue
        w   = weights.get(mod_name, 0.0)
        sc  = mod_res['score']
        eff = _effective_confidence(mod_res['confidence'], config)
        if eff > 0.0:
            if config.get('DEBUG', False):
                print(f"[DEBUG] {mod_name}: score={sc} eff_conf={eff:.3f} w={w} contribution={sc*w*eff:.3f}")
            num += sc * w * eff  # type: ignore
            den += w * eff  # type: ignore
        else:
            if config.get('DEBUG', False):
                print(f"[DEBUG] {mod_name}: ZEROED OUT (eff_conf={eff:.3f} < threshold)")

    if den == 0.0:
        return None   # all modules skipped
    return num / (den + 1e-9)  # type: ignore


def aggregate(preflight: dict, frame_result: dict, face_group: dict, body_group: dict, config: dict) -> tuple:
    """
    STAGE 8: AGGREGATION
    LEVEL 2 — across groups, with weight redistribution for None groups.
    Applies a graduated penalty to the final score based on image sharpness (Laplacian variance)
    to prevent blurry images with perfect poses from ranking artificially high.
    Returns (final_score, score_band).
    """
    frame_score = frame_result.get('offset_score', 0.0)

    # ── Use pre-computed group scores (penalties already baked in) ───────────
    face_score = face_group.get('group_score')
    if face_score is None:
        face_score = _aggregate_group(
            face_group.get('modules', {}), config['FACE_WEIGHTS'], config
        )
        face_group['group_score'] = round(face_score, 1) if face_score is not None else None  # type: ignore

    body_score = body_group.get('group_score')
    if body_score is None:
        body_score = _aggregate_group(
            body_group.get('modules', {}), config['BODY_WEIGHTS'], config
        )
        body_group['group_score'] = round(body_score, 1) if body_score is not None else None  # type: ignore

    # Keep DEBUG prints for module contribution visibility (secondary pass, no side-effects)
    _aggregate_group(face_group.get('modules', {}), config['FACE_WEIGHTS'], config)
    _aggregate_group(body_group.get('modules', {}), config['BODY_WEIGHTS'], config)

    # LEVEL 2 — only include non-None groups
    active: dict[str, float] = {'frame_offset': frame_score}
    if face_score is not None:
        active['face_group'] = face_score
    if body_score is not None:
        active['body_group'] = body_score

    total_w = sum(config['GROUP_WEIGHTS'][k] for k in active)
    if total_w == 0.0:
        return None, ""

    final = sum(active[k] * config['GROUP_WEIGHTS'][k] for k in active) / (total_w + 1e-9)

    # ── SHARPNESS PENALTY MULTIPLIER ─────────────────────────────────────────
    # We penalize frames that barely passed the blur_threshold. Blurry images
    # should NEVER outrank sharp images, even if the person's pose is flawless.
    blur_score = preflight.get('blur_score', 500.0)
    blur_threshold = config['PREFLIGHT']['blur_threshold']
    
    # safe_cap is the Laplacian variance where no penalty is applied (1.0x).
    # Since webcams vary wildly, we assign safe_cap dynamically 1.5x above the 
    # minimum required threshold.
    safe_cap = max(150.0, blur_threshold + (blur_threshold * 0.5))
    
    if blur_score < safe_cap and blur_threshold > 0:
        # Scale between 0.0 (at exactly threshold) to 1.0 (at safe_cap)
        ratio = max(0.0, (blur_score - blur_threshold) / (safe_cap - blur_threshold))
        
        # A frame that perfectly rides the threshold (e.g. 85.1 blur_score) gets
        # cut safely in half: 0.50x multiplier.
        multiplier = 0.50 + (0.50 * ratio)
        
        if config.get('DEBUG', False):
            print(f"[SHARPNESS PENALTY] Blur:{blur_score:.1f} SafeCap:{safe_cap:.1f} -> Mutliplier: {multiplier:.2f}x (Score {final:.1f} -> {final*multiplier:.1f})")
            
        final *= multiplier

    final = round(final, 2)  # type: ignore
    return final, _score_band(final, config)

