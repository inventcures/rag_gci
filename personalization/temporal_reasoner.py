"""
Temporal Reasoner

Performs temporal analysis on patient observations to:
- Detect trends (improving, stable, worsening)
- Identify patterns (diurnal, seasonal, treatment-related)
- Correlate medication changes with symptom changes
- Generate progression reports

This enables proactive care by identifying concerning patterns
before they become emergencies.
"""

import logging
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from enum import Enum
from statistics import mean, stdev
import asyncio

from .longitudinal_memory import (
    LongitudinalPatientRecord,
    LongitudinalMemoryManager,
    TimestampedObservation,
    SymptomObservation,
    MedicationEvent,
    SeverityLevel,
    TemporalTrend,
    MedicationAction,
)

logger = logging.getLogger(__name__)


# ============================================================================
# TEMPORAL ANALYSIS RESULTS
# ============================================================================

@dataclass
class SymptomProgressionReport:
    """
    Detailed analysis of how a symptom has changed over time.

    Provides insights for clinical decision-making about:
    - Disease progression
    - Treatment effectiveness
    - Need for intervention
    """
    symptom_name: str
    patient_id: str
    analysis_period_days: int
    generated_at: datetime = field(default_factory=datetime.now)

    # Data summary
    total_observations: int = 0
    date_range: Optional[Tuple[datetime, datetime]] = None

    # Current state
    current_severity: SeverityLevel = SeverityLevel.MODERATE
    baseline_severity: SeverityLevel = SeverityLevel.MODERATE
    latest_observation_date: Optional[datetime] = None

    # Temporal analysis
    trend: TemporalTrend = TemporalTrend.UNKNOWN
    trend_confidence: float = 0.0  # R-squared
    trend_description: str = ""

    # Pattern detection
    diurnal_pattern: Optional[str] = None  # "worse_morning", "worse_evening", "none"
    weekly_pattern: Optional[str] = None  # "weekday_trend", "weekend_trend", "none"

    # Rate of change
    severity_change_per_week: float = 0.0
    estimated_days_to_next_level: Optional[int] = None

    # Correlations
    response_to_medication: Optional[str] = None  # "improves_with", "no_change", "worsens_with"
    correlated_medications: List[str] = field(default_factory=list)

    # Clinical interpretation
    clinical_concerns: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)

    # Visualization data
    time_series_data: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "symptom_name": self.symptom_name,
            "patient_id": self.patient_id,
            "analysis_period_days": self.analysis_period_days,
            "generated_at": self.generated_at.isoformat(),
            "total_observations": self.total_observations,
            "date_range": [
                self.date_range[0].isoformat(),
                self.date_range[1].isoformat()
            ] if self.date_range else None,
            "current_severity": self.current_severity.value,
            "baseline_severity": self.baseline_severity.value,
            "latest_observation_date": self.latest_observation_date.isoformat() if self.latest_observation_date else None,
            "trend": self.trend.value,
            "trend_confidence": self.trend_confidence,
            "trend_description": self.trend_description,
            "diurnal_pattern": self.diurnal_pattern,
            "weekly_pattern": self.weekly_pattern,
            "severity_change_per_week": self.severity_change_per_week,
            "estimated_days_to_next_level": self.estimated_days_to_next_level,
            "response_to_medication": self.response_to_medication,
            "correlated_medications": self.correlated_medications,
            "clinical_concerns": self.clinical_concerns,
            "recommended_actions": self.recommended_actions,
            "time_series_data": self.time_series_data
        }


@dataclass
class MedicationEffectivenessReport:
    """
    Analysis of medication effectiveness over time.

    Helps answer:
    - Is this medication working?
    - Should we consider rotation?
    - Are there side effects?
    """
    medication_name: str
    patient_id: str
    analysis_period_days: int
    generated_at: datetime = field(default_factory=datetime.now)

    # Usage data
    total_doses_recorded: int = 0
    adherence_rate: float = 0.0  # 0.0 to 1.0
    missed_doses: int = 0

    # Effectiveness tracking
    effectiveness_trend: TemporalTrend = TemporalTrend.UNKNOWN
    associated_symptoms: List[str] = field(default_factory=list)
    symptom_improvement: Dict[str, str] = field(default_factory=dict)

    # Side effects
    reported_side_effects: List[str] = field(default_factory=list)
    side_effect_severity: SeverityLevel = SeverityLevel.NONE

    # Recommendations
    should_consider_rotation: bool = False
    rotation_reason: str = ""
    recommended_actions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "medication_name": self.medication_name,
            "patient_id": self.patient_id,
            "analysis_period_days": self.analysis_period_days,
            "generated_at": self.generated_at.isoformat(),
            "total_doses_recorded": self.total_doses_recorded,
            "adherence_rate": self.adherence_rate,
            "missed_doses": self.missed_doses,
            "effectiveness_trend": self.effectiveness_trend.value,
            "associated_symptoms": self.associated_symptoms,
            "symptom_improvement": self.symptom_improvement,
            "reported_side_effects": self.reported_side_effects,
            "side_effect_severity": self.side_effect_severity.value,
            "should_consider_rotation": self.should_consider_rotation,
            "rotation_reason": self.rotation_reason,
            "recommended_actions": self.recommended_actions
        }


@dataclass
class CorrelationAnalysis:
    """
    Analysis of correlations between different observations.

    Identifies relationships like:
    - "Pain worsens when X medication is missed"
    - "Nausea improves after taking Y medication"
    """
    variable_1: str  # e.g., "pain"
    variable_2: str  # e.g., "morphine"
    correlation_type: str  # "positive", "negative", "none"
    correlation_strength: float  # -1.0 to 1.0
    confidence: float = 0.0
    description: str = ""
    clinical_significance: str = ""  # "significant", "moderate", "minimal"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "variable_1": self.variable_1,
            "variable_2": self.variable_2,
            "correlation_type": self.correlation_type,
            "correlation_strength": self.correlation_strength,
            "confidence": self.confidence,
            "description": self.description,
            "clinical_significance": self.clinical_significance
        }


# ============================================================================
# TEMPORAL REASONER
# ============================================================================

class TemporalReasoner:
    """
    Performs temporal reasoning on patient observations.

    Key capabilities:
    1. Trend detection with confidence scores
    2. Pattern recognition (diurnal, weekly)
    3. Medication effectiveness analysis
    4. Correlation detection
    5. Progression reporting
    """

    def __init__(
        self,
        longitudinal_manager: LongitudinalMemoryManager
    ):
        """
        Initialize the temporal reasoner.

        Args:
            longitudinal_manager: Manager for longitudinal patient records
        """
        self.longitudinal = longitudinal_manager

        logger.info("TemporalReasoner initialized")

    async def analyze_symptom_progression(
        self,
        patient_id: str,
        symptom_name: str,
        time_window_days: int = 90
    ) -> Optional[SymptomProgressionReport]:
        """
        Analyze how a symptom has progressed over time.

        Args:
            patient_id: Patient identifier
            symptom_name: Symptom to analyze
            time_window_days: Number of days to look back

        Returns:
            Detailed progression report or None if insufficient data
        """
        record = await self.longitudinal.get_or_create_record(patient_id)

        # Get observations for this symptom
        end_date = datetime.now()
        start_date = end_date - timedelta(days=time_window_days)

        observations = [
            o for o in record.observations
            if o.category == "symptom"
            and o.entity_name.lower() == symptom_name.lower()
            and start_date <= o.timestamp <= end_date
        ]

        if len(observations) < 3:
            logger.debug(f"Insufficient data for symptom progression: {symptom_name}")
            return None

        # Sort by timestamp
        observations.sort(key=lambda o: o.timestamp)

        # Extract severity values
        severity_values = []
        for obs in observations:
            if isinstance(obs, SymptomObservation):
                severity_values.append(obs.severity)
            elif isinstance(obs.value, SeverityLevel):
                severity_values.append(obs.value)
            elif isinstance(obs.value, (int, float)):
                severity_values.append(SeverityLevel(min(4, max(0, int(obs.value)))))
            else:
                severity_values.append(SeverityLevel.MODERATE)

        numeric_values = [s.value for s in severity_values]

        # Calculate trend
        trend, confidence = self._calculate_trend(observations, numeric_values)

        # Create report
        report = SymptomProgressionReport(
            symptom_name=symptom_name,
            patient_id=patient_id,
            analysis_period_days=time_window_days,
            total_observations=len(observations),
            date_range=(observations[0].timestamp, observations[-1].timestamp),
            current_severity=severity_values[-1],
            baseline_severity=severity_values[0],
            latest_observation_date=observations[-1].timestamp,
            trend=trend,
            trend_confidence=confidence,
            trend_description=self._describe_trend(trend, confidence, symptom_name)
        )

        # Calculate rate of change
        if len(numeric_values) >= 2:
            change = numeric_values[-1] - numeric_values[0]
            report.severity_change_per_week = change / (time_window_days / 7)

        # Estimate time to next severity level
        if trend == TemporalTrend.WORSENING and report.severity_change_per_week > 0:
            current_val = numeric_values[-1]
            next_level = min(4, current_val + 1)
            weeks_to_next = (next_level - current_val) / report.severity_change_per_week if report.severity_change_per_week > 0 else None
            if weeks_to_next and weeks_to_next > 0:
                report.estimated_days_to_next_level = int(weeks_to_next * 7)

        # Detect patterns
        report.diurnal_pattern = self._detect_diurnal_pattern(observations)
        report.weekly_pattern = self._detect_weekly_pattern(observations)

        # Analyze medication response
        report.response_to_medication, report.correlated_medications = await self._analyze_medication_response(
            patient_id, symptom_name, start_date, end_date
        )

        # Generate clinical concerns
        report.clinical_concerns = self._generate_clinical_concerns(report)

        # Generate recommendations
        report.recommended_actions = self._generate_recommendations(report)

        # Prepare time series data for visualization
        report.time_series_data = [
            {
                "timestamp": obs.timestamp.isoformat(),
                "severity": severity_values[i].value,
                "value_text": obs.value_text
            }
            for i, obs in enumerate(observations)
        ]

        return report

    async def analyze_medication_effectiveness(
        self,
        patient_id: str,
        medication_name: str,
        time_window_days: int = 90
    ) -> Optional[MedicationEffectivenessReport]:
        """
        Analyze if a medication is working effectively.

        Args:
            patient_id: Patient identifier
            medication_name: Medication to analyze
            time_window_days: Number of days to look back

        Returns:
            Medication effectiveness report or None if insufficient data
        """
        record = await self.longitudinal.get_or_create_record(patient_id)

        # Get medication events
        end_date = datetime.now()
        start_date = end_date - timedelta(days=time_window_days)

        med_events = [
            o for o in record.observations
            if o.category == "medication"
            and o.entity_name.lower() == medication_name.lower()
            and start_date <= o.timestamp <= end_date
        ]

        if not med_events:
            return None

        # Count adherence
        total_doses = 0
        missed_doses = 0
        for event in med_events:
            if isinstance(event, MedicationEvent):
                if event.action == MedicationAction.TAKEN:
                    total_doses += 1
                elif event.action == MedicationAction.MISSED:
                    missed_doses += 1
                    total_doses += 1
                elif event.action in [MedicationAction.STARTED, MedicationAction.DOSE_CHANGED]:
                    total_doses += 1

        adherence_rate = 1.0 - (missed_doses / total_doses) if total_doses > 0 else 0.0

        # Find symptoms that might be affected by this medication
        associated_symptoms = []
        symptom_improvement = {}

        # Common symptom-medication associations
        symptom_medication_map = {
            "morphine": ["pain", "breathlessness"],
            "fentanyl": ["pain"],
            "oxycodone": ["pain"],
            "tramadol": ["pain"],
            "ondansetron": ["nausea", "vomiting"],
            "metoclopramide": ["nausea", "vomiting"],
            "lactulose": ["constipation"],
            "bisacodyl": ["constipation"],
            "lorazepam": ["anxiety", "insomnia"],
            "dexamethasone": ["pain", "appetite_loss", "nausea"],
            "prednisolone": ["pain", "appetite_loss"]
        }

        potential_symptoms = symptom_medication_map.get(medication_name.lower(), [])

        for symptom in potential_symptoms:
            symptom_obs = [
                o for o in record.observations
                if o.category == "symptom"
                and o.entity_name.lower() == symptom.lower()
                and start_date <= o.timestamp <= end_date
            ]

            if symptom_obs:
                associated_symptoms.append(symptom)

                # Check trend for this symptom
                _, trend = self._calculate_trend_from_observations(symptom_obs)
                symptom_improvement[symptom] = trend.value

        # Determine effectiveness trend
        improving_count = sum(1 for t in symptom_improvement.values() if t == "improving")
        worsening_count = sum(1 for t in symptom_improvement.values() if t == "worsening")

        if improving_count > worsening_count:
            effectiveness_trend = TemporalTrend.IMPROVING
        elif worsening_count > improving_count:
            effectiveness_trend = TemporalTrend.WORSENING
        else:
            effectiveness_trend = TemporalTrend.STABLE

        # Check for side effects
        side_effects = []
        for event in med_events:
            if isinstance(event, MedicationEvent):
                side_effects.extend(event.side_effects)

        # Determine if rotation should be considered
        should_rotate = False
        rotation_reason = ""

        if effectiveness_trend == TemporalTrend.WORSENING and adherence_rate > 0.7:
            should_rotate = True
            rotation_reason = "Symptoms worsening despite good adherence"
        elif adherence_rate < 0.5:
            should_rotate = True
            rotation_reason = "Poor adherence - consider alternative formulation"

        report = MedicationEffectivenessReport(
            medication_name=medication_name,
            patient_id=patient_id,
            analysis_period_days=time_window_days,
            total_doses_recorded=total_doses,
            adherence_rate=adherence_rate,
            missed_doses=missed_doses,
            effectiveness_trend=effectiveness_trend,
            associated_symptoms=associated_symptoms,
            symptom_improvement=symptom_improvement,
            reported_side_effects=list(set(side_effects)),
            should_consider_rotation=should_rotate,
            rotation_reason=rotation_reason
        )

        # Generate recommendations
        if should_rotate:
            report.recommended_actions.append("Consider opioid rotation or alternative pain management")
        elif adherence_rate < 0.8:
            report.recommended_actions.append("Address barriers to adherence with patient")
        elif effectiveness_trend == TemporalTrend.IMPROVING:
            report.recommended_actions.append("Current regimen appears effective - continue")

        if side_effects:
            report.recommended_actions.append(f"Monitor for side effects: {', '.join(set(side_effects))}")

        return report

    async def find_correlations(
        self,
        patient_id: str,
        time_window_days: int = 90
    ) -> List[CorrelationAnalysis]:
        """
        Find correlations between medications and symptoms.

        Args:
            patient_id: Patient identifier
            time_window_days: Number of days to analyze

        Returns:
            List of detected correlations
        """
        record = await self.longitudinal.get_or_create_record(patient_id)

        end_date = datetime.now()
        start_date = end_date - timedelta(days=time_window_days)

        # Group observations by day for correlation analysis
        daily_data = self._build_daily_data(record, start_date, end_date)

        correlations = []

        # Check medication-symptom correlations
        medications = set()
        symptoms = set()

        for day_data in daily_data.values():
            medications.update(day_data.get("medications", []))
            symptoms.update(day_data.get("symptoms", []))

        for med in medications:
            for symptom in symptoms:
                correlation = self._calculate_medication_symptom_correlation(
                    daily_data, med, symptom
                )
                if correlation and abs(correlation.correlation_strength) > 0.3:
                    correlations.append(correlation)

        # Sort by strength
        correlations.sort(key=lambda c: abs(c.correlation_strength), reverse=True)

        return correlations[:10]  # Return top 10 correlations

    def _calculate_trend(
        self,
        observations: List[TimestampedObservation],
        numeric_values: List[float]
    ) -> Tuple[TemporalTrend, float]:
        """Calculate trend using linear regression."""
        if len(observations) < 3:
            return TemporalTrend.UNKNOWN, 0.0

        if len(numeric_values) < 3:
            return TemporalTrend.UNKNOWN, 0.0

        # Linear regression
        n = len(numeric_values)
        x = list(range(n))
        y = numeric_values

        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi * xi for xi in x)

        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return TemporalTrend.STABLE, 0.0

        slope = (n * sum_xy - sum_x * sum_y) / denominator

        # Calculate R-squared
        y_mean = sum_y / n
        ss_tot = sum((yi - y_mean) ** 2 for yi in y)

        if ss_tot == 0:
            return TemporalTrend.STABLE, 1.0

        y_pred = [slope * xi + (sum_y - slope * sum_x) / n for xi in x]
        ss_res = sum((yi - ypi) ** 2 for yi, ypi in zip(y, y_pred))

        r_squared = 1 - (ss_res / ss_tot)
        r_squared = max(0.0, min(1.0, r_squared))

        # Determine trend direction
        # For symptoms: positive slope = worsening
        if slope > 0.3:
            trend = TemporalTrend.WORSENING
        elif slope < -0.3:
            trend = TemporalTrend.IMPROVING
        elif abs(slope) < 0.15:
            trend = TemporalTrend.STABLE
        else:
            trend = TemporalTrend.FLUCTUATING

        return trend, r_squared

    def _calculate_trend_from_observations(
        self,
        observations: List[TimestampedObservation]
    ) -> Tuple[List[float], TemporalTrend]:
        """Calculate trend from observations directly."""
        numeric_values = []

        for obs in observations:
            if isinstance(obs.value, (int, float)):
                numeric_values.append(float(obs.value))
            elif isinstance(obs.value, SeverityLevel):
                numeric_values.append(float(obs.value.value))
            elif isinstance(obs, SymptomObservation):
                numeric_values.append(float(obs.severity.value))
            else:
                numeric_values.append(2.0)  # Default to moderate

        if len(numeric_values) < 3:
            return numeric_values, TemporalTrend.UNKNOWN

        # Simple trend calculation
        n = len(numeric_values)
        x = list(range(n))
        y = numeric_values

        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi * xi for xi in x)

        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return numeric_values, TemporalTrend.STABLE

        slope = (n * sum_xy - sum_x * sum_y) / denominator

        if slope > 0.3:
            return numeric_values, TemporalTrend.WORSENING
        elif slope < -0.3:
            return numeric_values, TemporalTrend.IMPROVING
        elif abs(slope) < 0.15:
            return numeric_values, TemporalTrend.STABLE
        else:
            return numeric_values, TemporalTrend.FLUCTUATING

    def _describe_trend(
        self,
        trend: TemporalTrend,
        confidence: float,
        symptom: str
    ) -> str:
        """Generate human-readable trend description."""
        confidence_desc = "high" if confidence > 0.7 else "moderate" if confidence > 0.4 else "low"

        descriptions = {
            TemporalTrend.IMPROVING: f"{symptom} has been improving ({confidence_desc} confidence)",
            TemporalTrend.STABLE: f"{symptom} has remained stable ({confidence_desc} confidence)",
            TemporalTrend.WORSENING: f"{symptom} has been worsening ({confidence_desc} confidence)",
            TemporalTrend.FLUCTUATING: f"{symptom} has been fluctuating ({confidence_desc} confidence)",
            TemporalTrend.UNKNOWN: f"Unable to determine trend for {symptom} (insufficient data)"
        }

        return descriptions.get(trend, descriptions[TemporalTrend.UNKNOWN])

    def _detect_diurnal_pattern(
        self,
        observations: List[TimestampedObservation]
    ) -> Optional[str]:
        """Detect if symptom follows a daily pattern."""
        if len(observations) < 4:
            return None

        # Group by time of day
        morning_values = []  # 6-12
        afternoon_values = []  # 12-18
        evening_values = []  # 18-24
        night_values = []  # 0-6

        for obs in observations:
            hour = obs.timestamp.hour

            severity = SeverityLevel.MODERATE
            if isinstance(obs, SymptomObservation):
                severity = obs.severity
            elif isinstance(obs.value, SeverityLevel):
                severity = obs.value

            if 6 <= hour < 12:
                morning_values.append(severity.value)
            elif 12 <= hour < 18:
                afternoon_values.append(severity.value)
            elif 18 <= hour < 24:
                evening_values.append(severity.value)
            else:
                night_values.append(severity.value)

        # Check for significant differences
        time_periods = {
            "morning": morning_values,
            "afternoon": afternoon_values,
            "evening": evening_values,
            "night": night_values
        }

        # Only analyze periods with data
        valid_periods = {k: v for k, v in time_periods.items() if v}

        if len(valid_periods) < 2:
            return None

        # Find highest and lowest average severity
        avg_by_period = {
            k: mean(v) for k, v in valid_periods.items()
        }

        max_period = max(avg_by_period, key=avg_by_period.get)
        min_period = min(avg_by_period, key=avg_by_period.get)

        # Check if difference is significant
        if avg_by_period[max_period] - avg_by_period[min_period] > 1.0:
            if max_period in ["morning", "afternoon"]:
                return "worse_daytime"
            elif max_period in ["evening", "night"]:
                return "worse_evening"

        return "none"

    def _detect_weekly_pattern(
        self,
        observations: List[TimestampedObservation]
    ) -> Optional[str]:
        """Detect if symptom follows a weekly pattern."""
        if len(observations) < 7:
            return None

        # Group by weekday
        weekday_values = {i: [] for i in range(7)}
        weekend_values = []

        for obs in observations:
            weekday = obs.timestamp.weekday()
            hour = obs.timestamp.hour

            severity = SeverityLevel.MODERATE
            if isinstance(obs, SymptomObservation):
                severity = obs.severity
            elif isinstance(obs.value, SeverityLevel):
                severity = obs.value

            weekday_values[weekday].append(severity.value)

            if weekday >= 5:  # Weekend
                weekend_values.append(severity.value)

        # Calculate averages
        weekday_avg = mean([v for vals in weekday_values.values() for v in vals])
        weekend_avg = mean(weekend_values) if weekend_values else weekday_avg

        # Check if difference is significant
        if abs(weekday_avg - weekend_avg) > 1.0:
            return "worse_weekend" if weekend_avg > weekday_avg else "worse_weekday"

        return "none"

    async def _analyze_medication_response(
        self,
        patient_id: str,
        symptom_name: str,
        start_date: datetime,
        end_date: datetime
    ) -> Tuple[Optional[str], List[str]]:
        """Analyze how symptom responds to medications."""
        record = await self.longitudinal.get_or_create_record(patient_id)

        # Get symptom observations
        symptom_obs = [
            o for o in record.observations
            if o.category == "symptom"
            and o.entity_name.lower() == symptom_name.lower()
            and start_date <= o.timestamp <= end_date
        ]

        # Get medication observations
        med_obs = [
            o for o in record.observations
            if o.category == "medication"
            and start_date <= o.timestamp <= end_date
        ]

        if not symptom_obs or not med_obs:
            return None, []

        # Extract medications taken
        medications_taken = set()
        for obs in med_obs:
            if isinstance(obs, MedicationEvent):
                if obs.action in [MedicationAction.TAKEN, MedicationAction.STARTED]:
                    medications_taken.add(obs.medication_name.lower())
            elif obs.entity_name:
                medications_taken.add(obs.entity_name.lower())

        return "unknown_response", list(medications_taken)

    def _generate_clinical_concerns(
        self,
        report: SymptomProgressionReport
    ) -> List[str]:
        """Generate clinical concerns based on progression report."""
        concerns = []

        # Worsening trend
        if report.trend == TemporalTrend.WORSENING and report.trend_confidence > 0.5:
            concerns.append(f"{report.symptom_name} shows worsening trend")

            # Check severity level
            if report.current_severity.value >= 3:
                concerns.append(f"{report.symptom_name} is currently severe")

            # Check rate of worsening
            if report.severity_change_per_week > 0.5:
                concerns.append(f"{report.symptom_name} is worsening rapidly ({report.severity_change_per_week:.1f} levels/week)")

        # High severity even if stable
        elif report.current_severity.value >= 3 and report.trend == TemporalTrend.STABLE:
            concerns.append(f"{report.symptom_name} remains severe despite stable trend")

        # No improvement with medication
        if report.response_to_medication == "no_change":
            concerns.append(f"{report.symptom_name} not responding to current medications")

        # Long time since last observation
        if report.latest_observation_date:
            days_since = (datetime.now() - report.latest_observation_date).days
            if days_since > 30:
                concerns.append(f"No {report.symptom_name} update in {days_since} days")

        return concerns

    def _generate_recommendations(
        self,
        report: SymptomProgressionReport
    ) -> List[str]:
        """Generate recommendations based on progression report."""
        recommendations = []

        if report.trend == TemporalTrend.WORSENING:
            recommendations.append(f"Reassess {report.symptom_name} management plan")
            recommendations.append("Consider clinical review")

            if report.current_severity.value >= 3:
                recommendations.append("Urgent assessment may be warranted")

        elif report.trend == TemporalTrend.IMPROVING:
            recommendations.append(f"Current {report.symptom_name} management appears effective")
            recommendations.append("Continue current approach")

        elif report.trend == TemporalTrend.STABLE:
            if report.current_severity.value <= 2:
                recommendations.append(f"{report.symptom_name} well controlled - continue monitoring")
            else:
                recommendations.append(f"Consider {report.symptom_name} management review")

        # Medication-related recommendations
        if report.correlated_medications:
            recommendations.append(f"Review effectiveness of: {', '.join(report.correlated_medications)}")

        # Pattern-based recommendations
        if report.diurnal_pattern == "worse_evening":
            recommendations.append("Consider evening dose adjustment")

        return recommendations

    def _build_daily_data(
        self,
        record: LongitudinalPatientRecord,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[date, Dict[str, Any]]:
        """Build daily aggregated data for correlation analysis."""
        daily_data = {}

        for obs in record.observations:
            if not (start_date <= obs.timestamp <= end_date):
                continue

            day = obs.timestamp.date()

            if day not in daily_data:
                daily_data[day] = {
                    "symptoms": [],
                    "medications": [],
                    "symptom_severity": {},
                    "medication_taken": {}
                }

            if obs.category == "symptom":
                if obs.entity_name not in daily_data[day]["symptoms"]:
                    daily_data[day]["symptoms"].append(obs.entity_name)

                severity = SeverityLevel.MODERATE
                if isinstance(obs, SymptomObservation):
                    severity = obs.severity
                elif isinstance(obs.value, SeverityLevel):
                    severity = obs.value

                daily_data[day]["symptom_severity"][obs.entity_name] = severity.value

            elif obs.category == "medication":
                if obs.entity_name not in daily_data[day]["medications"]:
                    daily_data[day]["medications"].append(obs.entity_name)

                taken = isinstance(obs, MedicationEvent) and obs.action == MedicationAction.TAKEN
                daily_data[day]["medication_taken"][obs.entity_name] = 1 if taken else 0

        return daily_data

    def _calculate_medication_symptom_correlation(
        self,
        daily_data: Dict[date, Dict[str, Any]],
        medication: str,
        symptom: str
    ) -> Optional[CorrelationAnalysis]:
        """Calculate correlation between medication and symptom."""
        # Build paired data
        paired = []
        for day_data in daily_data.values():
            med_taken = day_data["medication_taken"].get(medication, 0)
            symptom_severity = day_data["symptom_severity"].get(symptom)

            if symptom_severity is not None:
                paired.append((med_taken, symptom_severity))

        if len(paired) < 5:
            return None

        # Calculate correlation
        med_values = [p[0] for p in paired]
        symptom_values = [p[1] for p in paired]

        # Pearson correlation
        n = len(paired)
        sum_x = sum(med_values)
        sum_y = sum(symptom_values)
        sum_xy = sum(m * s for m, s in paired)
        sum_x2 = sum(m * m for m in med_values)
        sum_y2 = sum(s * s for s in symptom_values)

        denominator = (n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2)
        if denominator == 0:
            return None

        correlation = (n * sum_xy - sum_x * sum_y) / (denominator ** 0.5)

        # Determine significance
        strength = abs(correlation)
        if strength > 0.7:
            significance = "significant"
        elif strength > 0.4:
            significance = "moderate"
        else:
            significance = "minimal"

        # Determine type
        if correlation < -0.2:
            correlation_type = "negative"  # Medication reduces symptom (good)
            description = f"{medication} associated with reduced {symptom}"
        elif correlation > 0.2:
            correlation_type = "positive"  # Medication associated with increased symptom (side effect?)
            description = f"{medication} associated with increased {symptom}"
        else:
            correlation_type = "none"
            description = f"No clear correlation between {medication} and {symptom}"

        return CorrelationAnalysis(
            variable_1=medication,
            variable_2=symptom,
            correlation_type=correlation_type,
            correlation_strength=correlation,
            description=description,
            clinical_significance=significance
        )
