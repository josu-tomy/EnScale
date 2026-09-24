"""
Constrained equipment scheduling and energy optimization service for EnScale.

Optimization Strategy:
- Lightweight deterministic constrained enumeration over operating hour candidates.
- Non-flexible equipment (is_flexible == False) remains strictly unchanged.
- For flexible equipment (is_flexible == True):
    1. Enumerate feasible daily runtime candidates h in [minimum_hours, min(maximum_hours, window)]
    2. Calculate energy: kWh = rated_power_kw * quantity * h * operating_days * utilization_factor
    3. Calculate cost: INR = kWh * tariff_inr_per_kwh
    4. Reject invalid schedules violating runtime or building operating window bounds
    5. Select minimum-cost feasible schedule
- Zero operational constraints violated.
- Strictly modeled savings:
    energy_savings_kwh = baseline_energy_kwh - optimized_energy_kwh
    cost_savings_inr = baseline_cost_inr - optimized_cost_inr
    If savings <= 0: "No modeled savings under the current constraints."
"""

from typing import List, Dict, Any, Optional, Union
import numpy as np

from domain.models import BuildingProfile, Equipment, OptimizationResult
from config.constants import DEFAULT_TARIFF_INR_PER_KWH, DEFAULT_GRID_EMISSION_FACTOR_KG_PER_KWH


class InfeasibleConstraintError(ValueError):
    """Raised when equipment scheduling operational constraints cannot be satisfied."""
    pass


def compute_modeled_savings(
    baseline_energy_kwh: float,
    optimized_energy_kwh: float,
    tariff_inr_per_kwh: float,
) -> OptimizationResult:
    """
    Computes modeled savings according to the Canonical Savings Contract.
    Note: These reflect 'Projected savings under the modeled operating constraints'.
    Never guaranteed savings.
    """
    energy_savings_kwh = float(baseline_energy_kwh - optimized_energy_kwh)
    baseline_cost_inr = float(baseline_energy_kwh * tariff_inr_per_kwh)
    optimized_cost_inr = float(optimized_energy_kwh * tariff_inr_per_kwh)
    cost_savings_inr = float(baseline_cost_inr - optimized_cost_inr)

    savings_pct = (
        float((energy_savings_kwh / baseline_energy_kwh) * 100.0)
        if baseline_energy_kwh > 0
        else 0.0
    )

    if cost_savings_inr <= 0:
        status_msg = "No modeled savings under the current constraints."
    else:
        status_msg = "Projected savings under the modeled operating constraints."

    return OptimizationResult(
        baseline_energy_kwh=round(float(baseline_energy_kwh), 2),
        optimized_energy_kwh=round(float(optimized_energy_kwh), 2),
        energy_savings_kwh=round(energy_savings_kwh, 2),
        baseline_cost_inr=round(baseline_cost_inr, 2),
        optimized_cost_inr=round(optimized_cost_inr, 2),
        cost_savings_inr=round(cost_savings_inr, 2),
        savings_pct=round(savings_pct, 2),
        savings_statement="Projected savings under the modeled operating constraints",
        is_feasible=True,
        status_message=status_msg,
    )


def optimize_equipment_schedule(
    building_profile: Union[BuildingProfile, Dict[str, Any]],
    equipment_list: List[Union[Equipment, Dict[str, Any]]],
    tariff_inr_per_kwh: Optional[float] = None,
    step_hours: float = 0.5,
    raise_on_infeasible: bool = True,
    emission_factor_kg_per_kwh: float = DEFAULT_GRID_EMISSION_FACTOR_KG_PER_KWH,
) -> OptimizationResult:
    """
    Performs deterministic constrained schedule optimization across equipment assets.

    Constraints enforced:
    - Building operating window: window_hours = operating_end - operating_start
    - Equipment runtime range: minimum_hours <= h <= maximum_hours
    - Feasible operating window cap: h <= window_hours (for daytime building equipment)
    - Flexibility: is_flexible == False remains unchanged; is_flexible == True optimized
    - Operational requirements: never violate minimum_hours to achieve savings

    If constraints are impossible (e.g. minimum_hours > maximum_hours or minimum_hours > window_hours),
    raises InfeasibleConstraintError or returns an infeasible OptimizationResult.
    """
    # 1. Parse building parameters
    if isinstance(building_profile, BuildingProfile):
        b_dict = building_profile.to_dict()
    else:
        b_dict = building_profile

    op_start = int(b_dict["operating_start"])
    op_end = int(b_dict["operating_end"])
    window_hours = float(op_end - op_start)

    if window_hours <= 0:
        msg = f"Impossible building operating window: operating_start ({op_start}) >= operating_end ({op_end})."
        if raise_on_infeasible:
            raise InfeasibleConstraintError(msg)
        return OptimizationResult(
            baseline_energy_kwh=0.0,
            optimized_energy_kwh=0.0,
            energy_savings_kwh=0.0,
            baseline_cost_inr=0.0,
            optimized_cost_inr=0.0,
            cost_savings_inr=0.0,
            savings_pct=0.0,
            is_feasible=False,
            status_message=msg,
        )

    if tariff_inr_per_kwh is None:
        tariff = float(b_dict.get("tariff_inr_per_kwh", DEFAULT_TARIFF_INR_PER_KWH))
    else:
        tariff = float(tariff_inr_per_kwh)

    total_baseline_kwh = 0.0
    total_optimized_kwh = 0.0
    recommendations = []

    # 2. Process each equipment item
    for item in equipment_list:
        if isinstance(item, Equipment):
            eq = item
        else:
            eq = Equipment.from_dict(item)

        power = eq.rated_power_kw
        qty = eq.quantity
        base_h = eq.hours_per_day
        days = eq.operating_days
        util = eq.utilization_factor
        min_h = eq.minimum_hours
        max_h = eq.maximum_hours
        flexible = eq.is_flexible

        # Baseline energy calculation
        base_kwh = power * qty * base_h * days * util
        total_baseline_kwh += base_kwh

        # Check for impossible / contradictory constraints
        if min_h > max_h:
            msg = (
                f"Impossible constraints for equipment '{eq.equipment_name}' ({eq.equipment_id}): "
                f"minimum_hours ({min_h}) exceeds maximum_hours ({max_h}). No feasible solution."
            )
            if raise_on_infeasible:
                raise InfeasibleConstraintError(msg)
            return OptimizationResult(
                baseline_energy_kwh=round(total_baseline_kwh, 2),
                optimized_energy_kwh=round(total_baseline_kwh, 2),
                energy_savings_kwh=0.0,
                baseline_cost_inr=round(total_baseline_kwh * tariff, 2),
                optimized_cost_inr=round(total_baseline_kwh * tariff, 2),
                cost_savings_inr=0.0,
                savings_pct=0.0,
                is_feasible=False,
                status_message=msg,
            )

        if min_h > window_hours:
            msg = (
                f"Impossible constraints for equipment '{eq.equipment_name}' ({eq.equipment_id}): "
                f"minimum runtime ({min_h}h) exceeds building operating window ({window_hours}h). No feasible solution."
            )
            if raise_on_infeasible:
                raise InfeasibleConstraintError(msg)
            return OptimizationResult(
                baseline_energy_kwh=round(total_baseline_kwh, 2),
                optimized_energy_kwh=round(total_baseline_kwh, 2),
                energy_savings_kwh=0.0,
                baseline_cost_inr=round(total_baseline_kwh * tariff, 2),
                optimized_cost_inr=round(total_baseline_kwh * tariff, 2),
                cost_savings_inr=0.0,
                savings_pct=0.0,
                is_feasible=False,
                status_message=msg,
            )

        if not flexible:
            # Non-flexible equipment remains strictly unchanged
            opt_h = base_h
            opt_kwh = base_kwh
            recommendations.append({
                "equipment_id": eq.equipment_id,
                "equipment_name": eq.equipment_name,
                "is_flexible": False,
                "needed_until_closing": bool(getattr(eq, "needed_until_closing", False)),
                "binding_constraint": "Non-flexible equipment constraint (runtime cannot be altered)",
                "baseline_hours_per_day": base_h,
                "optimized_hours_per_day": opt_h,
                "hours_per_day": opt_h,
                "original_hours_per_day": base_h,
                "current_start_time": f"{op_start:02d}:00",
                "current_stop_time": f"{min(24, int(op_start + base_h)):02d}:00",
                "optimized_start_time": f"{op_start:02d}:00",
                "optimized_stop_time": f"{min(24, int(op_start + opt_h)):02d}:00",
                "operating_start": op_start,
                "operating_end": op_end,
                "minimum_hours": min_h,
                "maximum_hours": max_h,
                "baseline_energy_kwh": round(base_kwh, 2),
                "optimized_energy_kwh": round(opt_kwh, 2),
                "estimated_energy_kwh": round(opt_kwh, 2),
                "cost_inr": round(opt_kwh * tariff, 2),
                "modeled_energy_savings_kwh": 0.0,
                "modeled_cost_savings_inr": 0.0,
                "annual_energy_savings_kwh": 0.0,
                "annual_cost_savings_inr": 0.0,
                "annual_co2_savings_kg": 0.0,
                "status": "Non-flexible equipment unchanged",
            })
            total_optimized_kwh += opt_kwh
            continue

        # Flexible equipment: deterministic constrained enumeration
        # Upper bound cannot exceed maximum_hours or building operating window
        upper_limit = min(max_h, window_hours)
        if upper_limit < min_h:
            msg = (
                f"Upper limit ({upper_limit}h) is less than minimum_hours ({min_h}h) "
                f"for '{eq.equipment_name}'. No feasible solution."
            )
            if raise_on_infeasible:
                raise InfeasibleConstraintError(msg)
            return OptimizationResult(
                baseline_energy_kwh=round(total_baseline_kwh, 2),
                optimized_energy_kwh=round(total_baseline_kwh, 2),
                energy_savings_kwh=0.0,
                baseline_cost_inr=round(total_baseline_kwh * tariff, 2),
                optimized_cost_inr=round(total_baseline_kwh * tariff, 2),
                cost_savings_inr=0.0,
                savings_pct=0.0,
                is_feasible=False,
                status_message=msg,
            )

        # Enumerate feasible candidates
        candidates = []
        curr = min_h
        while curr <= upper_limit + 1e-6:
            candidates.append(round(curr, 2))
            curr += step_hours
        if upper_limit not in candidates:
            candidates.append(round(upper_limit, 2))

        feasible_schedules = []
        for cand_h in candidates:
            # Reject invalid schedules violating bounds
            if not (min_h <= cand_h <= max_h and cand_h <= window_hours):
                continue
            cand_kwh = power * qty * cand_h * days * util
            cand_cost = cand_kwh * tariff
            feasible_schedules.append({
                "hours_per_day": cand_h,
                "energy_kwh": cand_kwh,
                "cost_inr": cand_cost,
            })

        if not feasible_schedules:
            msg = f"No feasible schedules found for '{eq.equipment_name}'. Constraints cannot be satisfied."
            if raise_on_infeasible:
                raise InfeasibleConstraintError(msg)
            opt_h = base_h
            opt_kwh = base_kwh
        else:
            # Select minimum-cost feasible schedule (primary: cost, secondary: energy)
            best_schedule = min(feasible_schedules, key=lambda s: (s["cost_inr"], s["energy_kwh"]))
            opt_h = best_schedule["hours_per_day"]
            opt_kwh = best_schedule["energy_kwh"]

        opt_cost = opt_kwh * tariff
        base_cost = base_kwh * tariff
        savings_kwh = base_kwh - opt_kwh
        savings_inr = base_cost - opt_cost

        needed_from_opening = bool(getattr(eq, "needed_from_opening", False))
        needed_until_closing = bool(getattr(eq, "needed_until_closing", False))
        is_pump = any(k in eq.equipment_type.lower() or k in eq.equipment_name.lower() for k in ["pump", "water"])
        is_chiller = any(k in eq.equipment_type.lower() or k in eq.equipment_name.lower() for k in ["chiller", "hvac", "cooling"])

        curr_start = f"{op_start:02d}:00"
        curr_stop = f"{min(24, int(op_start + base_h)):02d}:00"

        if needed_from_opening and needed_until_closing:
            opt_start_hr = op_start
            opt_stop_hr = op_end
            opt_h = float(op_end - op_start)
            opt_start_str = f"{opt_start_hr:02d}:00"
            opt_stop_str = f"{opt_stop_hr:02d}:00"
            binding_constraint = (
                f"Constrained to run from opening ({op_start:02d}:00) until closing ({op_end:02d}:00). "
                f"Total runtime fixed to operational window ({opt_h:.1f}h)."
            )
            status_desc = f"Scheduled from {opt_start_str} to {opt_stop_str} ({opt_h:.1f}h)."
        elif needed_from_opening:
            # Cannot start later than operating_start (FIX 4)
            opt_start_hr = op_start
            opt_stop_hr = min(24, int(op_start + opt_h))
            opt_start_str = f"{opt_start_hr:02d}:00"
            opt_stop_str = f"{opt_stop_hr:02d}:00"
            binding_constraint = (
                f"Limited by minimum runtime you entered: {min_h:.1f} h (runtime reduced from {base_h:.1f}h to {opt_h:.1f}h); "
                f"scheduled from facility opening ({op_start:02d}:00)."
            )
            status_desc = f"Scheduled from {opt_start_str} to {opt_stop_str} ({opt_h:.1f}h), running from facility opening."
        elif needed_until_closing:
            # Cannot stop before operating_end (FIX 2)
            opt_stop_hr = op_end
            opt_start_hr = max(op_start, int(op_end - opt_h))
            opt_start_str = f"{opt_start_hr:02d}:00"
            opt_stop_str = f"{opt_stop_hr:02d}:00"
            binding_constraint = (
                f"Limited by minimum runtime ({min_h:.1f}h); constrained to maintain operation "
                f"until facility closing ({op_end:02d}:00)."
            )
            status_desc = f"Rescheduled to {opt_start_str}–{opt_stop_str} ({opt_h:.1f}h), maintaining operation until closing."
        elif is_pump:
            # Under flat tariff: savings derive entirely from cutting runtime, not time-of-day shifting (FIX 2)
            opt_start_str = f"{op_start:02d}:00"
            opt_stop_str = f"{min(24, int(op_start + opt_h)):02d}:00"
            binding_constraint = (
                f"Limited by minimum runtime you entered: {min_h:.1f} h (the largest runtime reduction allowed by your constraints). "
                "Under a flat tariff, savings derive entirely from cutting runtime, not time-of-day shifting."
            )
            status_desc = f"Runtime reduced from {base_h:.1f}h to {opt_h:.1f}h. Under a flat tariff, all savings come from runtime reduction."
        else:
            opt_start_str = f"{op_start:02d}:00"
            opt_stop_hr = min(24, int(op_start + opt_h))
            opt_stop_str = f"{opt_stop_hr:02d}:00"
            binding_constraint = (
                f"Limited by minimum runtime you entered: {min_h:.1f} h (runtime reduced from {base_h:.1f}h to {opt_h:.1f}h)."
            )
            status_desc = f"Runtime reduced from {base_h:.1f}h to {opt_h:.1f}h (min required: {min_h:.1f}h)."

        # Warning when schedule start time changes (FIX 4)
        start_time_changed = (opt_start_str != curr_start)
        start_time_warning = "Comfort and process impact not modeled — verify on site." if start_time_changed else ""

        recommendations.append({
            "equipment_id": eq.equipment_id,
            "equipment_name": eq.equipment_name,
            "is_flexible": True,
            "needed_from_opening": needed_from_opening,
            "needed_until_closing": needed_until_closing,
            "start_time_changed": start_time_changed,
            "start_time_warning": start_time_warning,
            "binding_constraint": binding_constraint,
            "baseline_hours_per_day": base_h,
            "optimized_hours_per_day": opt_h,
            "hours_per_day": opt_h,
            "original_hours_per_day": base_h,
            "current_start_time": curr_start,
            "current_stop_time": curr_stop,
            "optimized_start_time": opt_start_str,
            "optimized_stop_time": opt_stop_str,
            "operating_start": op_start,
            "operating_end": op_end,
            "minimum_hours": min_h,
            "maximum_hours": max_h,
            "baseline_energy_kwh": round(base_kwh, 2),
            "optimized_energy_kwh": round(opt_kwh, 2),
            "estimated_energy_kwh": round(opt_kwh, 2),
            "cost_inr": round(opt_cost, 2),
            "modeled_energy_savings_kwh": round(savings_kwh, 2),
            "modeled_cost_savings_inr": round(savings_inr, 2),
            "annual_energy_savings_kwh": round(savings_kwh * 12.0, 1),
            "annual_cost_savings_inr": round(savings_inr * 12.0, 0),
            "annual_co2_savings_kg": round(savings_kwh * 12.0 * emission_factor_kg_per_kwh, 1),
            "status": status_desc if savings_kwh > 0 else "Operated at optimal minimum feasible runtime",
        })
        total_optimized_kwh += opt_kwh

    # 3. Compile overall optimization result
    res = compute_modeled_savings(
        baseline_energy_kwh=total_baseline_kwh,
        optimized_energy_kwh=total_optimized_kwh,
        tariff_inr_per_kwh=tariff,
    )
    res.schedule_recommendations = recommendations
    return res


class OptimizationService:
    """Service class encapsulating equipment schedule optimization."""

    def __init__(self, default_tariff_inr_per_kwh: float = DEFAULT_TARIFF_INR_PER_KWH):
        self.default_tariff = default_tariff_inr_per_kwh

    def optimize_equipment_schedule(
        self,
        building_profile: Union[BuildingProfile, Dict[str, Any]],
        equipment_list: List[Union[Equipment, Dict[str, Any]]],
        tariff_inr_per_kwh: Optional[float] = None,
    ) -> OptimizationResult:
        """
        Executes constrained equipment schedule optimization.
        """
        return optimize_equipment_schedule(
            building_profile=building_profile,
            equipment_list=equipment_list,
            tariff_inr_per_kwh=tariff_inr_per_kwh or self.default_tariff,
        )

    # Backward compatibility
    def optimize_schedule(
        self,
        equipment_list: List[Equipment],
        tariff_inr_per_kwh: float,
        target_reduction_pct: float = 10.0,
    ) -> OptimizationResult:
        default_profile = {
            "operating_start": 8,
            "operating_end": 20,
            "operating_days_per_month": 22,
            "tariff_inr_per_kwh": tariff_inr_per_kwh,
        }
        return optimize_equipment_schedule(
            building_profile=default_profile,
            equipment_list=equipment_list,
            tariff_inr_per_kwh=tariff_inr_per_kwh,
        )
