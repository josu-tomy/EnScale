"""
Incentive matching service for EnScale.

Matches building profile and equipment inventory against local reference incentive policies
(e.g., BEE star ratings, state utility demand-side management rebates).

NOTE: EnScale provides transparent modeled estimations; it is not an official government
eligibility authority.
"""

from typing import List, Dict, Any
from domain.models import BuildingProfile, Equipment, IncentiveMatch
from domain.enums import EligibilityStatus


class IncentiveService:
    """Service skeleton for matching reference energy efficiency incentives."""

    def match_incentives(
        self,
        building_profile: BuildingProfile,
        equipment_list: List[Equipment],
    ) -> List[IncentiveMatch]:
        """
        Evaluates applicable incentives based on equipment types and building characteristics.
        """
        matches: List[IncentiveMatch] = []

        equipment_types = {eq.equipment_type for eq in equipment_list}

        if "HVAC" in equipment_types:
            matches.append(
                IncentiveMatch(
                    incentive_id="INC-HVAC-BEE-01",
                    program_name="BEE High-Efficiency Chiller / VRF Retrofit Scheme",
                    target_equipment_type="HVAC",
                    eligibility_status=EligibilityStatus.CONDITIONALLY_ELIGIBLE.value,
                    estimated_rebate_inr=25000.0,
                    policy_reference="National Mission for Enhanced Energy Efficiency (NMEEE)",
                    description="Rebate for retrofitting with BEE 5-Star rated high-efficiency HVAC equipment.",
                )
            )

        if "Motors & Pumps" in equipment_types:
            matches.append(
                IncentiveMatch(
                    incentive_id="INC-MOT-IE4-02",
                    program_name="IE3/IE4 Premium Efficiency Motor Adoption Scheme",
                    target_equipment_type="Motors & Pumps",
                    eligibility_status=EligibilityStatus.ELIGIBLE.value,
                    estimated_rebate_inr=15000.0,
                    policy_reference="SIDBI Energy Efficiency Scheme for MSMEs",
                    description="Capital subsidy support for adopting IE3/IE4 energy-efficient electric motors.",
                )
            )

        return matches
