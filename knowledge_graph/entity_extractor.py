"""
Entity Extractor for Palliative Care Knowledge Graph

Extracts medical entities and relationships from documents using LLM and patterns.
Specifically designed for palliative care guidelines and common queries.

Entity categories:
- Symptoms (pain types, respiratory, GI, neurological, end-of-life)
- Medications (opioids, anti-emetics, anxiolytics, laxatives, etc.)
- Conditions (cancer, organ failure, neurological diseases)
- Treatments & Interventions (pharmacological and non-pharmacological)
- Care Settings (hospice, home care, hospital)
- Care Goals (comfort care, symptom control, quality of life)
- Assessments (pain scales, performance status)
- Routes of Administration (oral, SC, IV, transdermal)
- Psychosocial (grief, spiritual distress, caregiver burden)
- Advance Care Planning (DNR, advance directives, goals of care)
"""

import os
import json
import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class EntityType(Enum):
    """Types of entities in palliative care domain."""
    # Clinical entities
    SYMPTOM = "Symptom"
    MEDICATION = "Medication"
    CONDITION = "Condition"
    TREATMENT = "Treatment"
    PROCEDURE = "Procedure"
    SIDE_EFFECT = "SideEffect"
    BODY_PART = "BodyPart"

    # Dosing entities
    DOSAGE = "Dosage"
    FREQUENCY = "Frequency"
    ROUTE = "Route"

    # Palliative care specific
    CARE_SETTING = "CareSetting"
    CARE_GOAL = "CareGoal"
    ASSESSMENT_TOOL = "AssessmentTool"
    INTERVENTION = "Intervention"

    # Psychosocial
    PSYCHOSOCIAL = "Psychosocial"
    SPIRITUAL = "Spiritual"

    # Advance care planning
    ADVANCE_CARE = "AdvanceCarePlanning"

    # People/Roles
    CAREGIVER_ROLE = "CaregiverRole"
    PATIENT_POPULATION = "PatientPopulation"


class RelationshipType(Enum):
    """Types of relationships in palliative care domain."""
    # Treatment relationships
    TREATS = "TREATS"                      # Medication -> Symptom
    ALLEVIATES = "ALLEVIATES"             # Medication/Intervention -> Symptom
    MANAGES = "MANAGES"                    # Treatment -> Condition
    PREVENTS = "PREVENTS"                  # Medication -> SideEffect

    # Causal relationships
    CAUSES = "CAUSES"                      # Condition -> Symptom
    SIDE_EFFECT_OF = "SIDE_EFFECT_OF"     # SideEffect -> Medication
    TRIGGERS = "TRIGGERS"                  # Condition -> Symptom
    EXACERBATES = "EXACERBATES"           # Factor -> Symptom

    # Clinical relationships
    INDICATES = "INDICATES"                # Symptom -> Condition
    CONTRAINDICATES = "CONTRAINDICATES"   # Condition -> Medication
    REQUIRES = "REQUIRES"                  # Procedure -> Medication
    AFFECTS = "AFFECTS"                    # Condition -> BodyPart
    ASSESSED_BY = "ASSESSED_BY"           # Symptom -> AssessmentTool

    # Administration relationships
    ADMINISTERED_VIA = "ADMINISTERED_VIA"  # Medication -> Route
    DOSAGE_FOR = "DOSAGE_FOR"             # Dosage -> Medication

    # Care relationships
    APPROPRIATE_FOR = "APPROPRIATE_FOR"    # CareGoal -> Condition
    PROVIDED_IN = "PROVIDED_IN"            # Treatment -> CareSetting
    INVOLVES = "INVOLVES"                  # CareGoal -> Intervention
    SUPPORTS = "SUPPORTS"                  # Intervention -> Psychosocial


@dataclass
class Entity:
    """Represents an extracted entity."""
    name: str
    type: EntityType
    properties: Dict[str, Any] = field(default_factory=dict)
    source_text: str = ""
    confidence: float = 1.0
    aliases: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type.value,
            "properties": self.properties,
            "source_text": self.source_text,
            "confidence": self.confidence,
            "aliases": self.aliases
        }


@dataclass
class Relationship:
    """Represents an extracted relationship between entities."""
    source: str
    source_type: EntityType
    target: str
    target_type: EntityType
    relationship: RelationshipType
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "source_type": self.source_type.value,
            "target": self.target,
            "target_type": self.target_type.value,
            "relationship": self.relationship.value,
            "properties": self.properties,
            "confidence": self.confidence
        }


# Enhanced extraction prompt for LLM
ENTITY_EXTRACTION_PROMPT = """You are a medical entity extractor specializing in palliative care.
Extract entities and relationships from the given text.

Entity Types:
- Symptom: Pain (bone, neuropathic, visceral, breakthrough), nausea, vomiting, dyspnea, fatigue,
  anxiety, depression, constipation, delirium, agitation, death rattle, cachexia, anorexia, etc.
- Medication: Opioids (morphine, fentanyl, oxycodone), anti-emetics (ondansetron, metoclopramide),
  anxiolytics (midazolam, lorazepam), laxatives (bisacodyl, lactulose), steroids (dexamethasone), etc.
- Condition: Cancer, heart failure, COPD, dementia, renal failure, liver failure, HIV/AIDS, ALS, etc.
- Treatment: Palliative care, hospice care, symptom management, pain management, etc.
- Procedure: Paracentesis, thoracentesis, nerve block, palliative sedation, wound care, etc.
- CareSetting: Hospice, home care, hospital, nursing home, inpatient, outpatient
- CareGoal: Comfort care, symptom control, quality of life, dignity, peaceful death
- AssessmentTool: Pain scale, ECOG, PPS, Edmonton Symptom Assessment
- Route: Oral, subcutaneous (SC), intravenous (IV), transdermal, sublingual, rectal
- Psychosocial: Grief, bereavement, caregiver burden, anxiety, depression, isolation
- AdvanceCarePlanning: DNR, advance directive, living will, goals of care discussion, POLST

Relationship Types:
- TREATS: Medication treats symptom
- ALLEVIATES: Intervention alleviates symptom
- CAUSES: Condition causes symptom
- SIDE_EFFECT_OF: Side effect is caused by medication
- MANAGES: Treatment manages condition
- CONTRAINDICATES: Condition contraindicates medication
- ADMINISTERED_VIA: Medication administered via route
- ASSESSED_BY: Symptom assessed by tool
- APPROPRIATE_FOR: Care goal appropriate for condition

TEXT TO ANALYZE:
{text}

Respond in JSON format:
{{
  "entities": [
    {{"name": "entity name", "type": "EntityType", "properties": {{}}}}
  ],
  "relationships": [
    {{"source": "entity1", "source_type": "Type1", "target": "entity2", "target_type": "Type2", "relationship": "RELATIONSHIP_TYPE"}}
  ]
}}
"""


class EntityExtractor:
    """
    Extracts medical entities and relationships from palliative care text.

    Features:
    - Comprehensive palliative care entity patterns
    - LLM-based extraction for complex relationships
    - Multi-language support (patterns focus on English/Hindi medical terms)

    Usage:
        extractor = EntityExtractor()
        entities, relationships = await extractor.extract(text)
    """

    def __init__(
        self,
        llm_provider: str = "groq",
        use_patterns: bool = True
    ):
        """
        Initialize entity extractor.

        Args:
            llm_provider: LLM provider ("groq", "openai", or "none")
            use_patterns: Whether to also use pattern matching
        """
        self.llm_provider = llm_provider
        self.use_patterns = use_patterns

        # Initialize LLM client
        self._groq_client = None
        self._openai_client = None

        if llm_provider == "groq" and os.getenv("GROQ_API_KEY"):
            try:
                from groq import AsyncGroq
                self._groq_client = AsyncGroq()
            except ImportError:
                logger.warning("groq package not installed")

        elif llm_provider == "openai" and os.getenv("OPENAI_API_KEY"):
            try:
                from openai import AsyncOpenAI
                self._openai_client = AsyncOpenAI()
            except ImportError:
                logger.warning("openai package not installed")

        # Initialize comprehensive palliative care patterns
        self._init_symptom_patterns()
        self._init_medication_patterns()
        self._init_condition_patterns()
        self._init_care_patterns()
        self._init_assessment_patterns()
        self._init_route_patterns()
        self._init_psychosocial_patterns()
        self._init_intervention_patterns()

    def _init_symptom_patterns(self):
        """Initialize symptom patterns for palliative care."""
        self._symptom_patterns = {
            # Pain types
            "pain_types": [
                (r'\b(bone pain|skeletal pain)\b', "Bone Pain"),
                (r'\b(neuropathic pain|nerve pain)\b', "Neuropathic Pain"),
                (r'\b(visceral pain|abdominal pain)\b', "Visceral Pain"),
                (r'\b(breakthrough pain|incident pain)\b', "Breakthrough Pain"),
                (r'\b(chronic pain|persistent pain)\b', "Chronic Pain"),
                (r'\b(acute pain)\b', "Acute Pain"),
                (r'\b(cancer pain|malignant pain)\b', "Cancer Pain"),
                (r'\b(total pain)\b', "Total Pain"),
                (r'\b(pain|ache|discomfort|soreness)\b', "Pain"),
            ],
            # Respiratory symptoms
            "respiratory": [
                (r'\b(dyspnea|breathlessness|shortness of breath|difficulty breathing)\b', "Dyspnea"),
                (r'\b(death rattle|terminal secretions|noisy breathing)\b', "Death Rattle"),
                (r'\b(cough|persistent cough|chronic cough)\b', "Cough"),
                (r'\b(hemoptysis|coughing blood)\b', "Hemoptysis"),
                (r'\b(stridor|wheeze|wheezing)\b', "Stridor"),
                (r'\b(hypoxia|low oxygen)\b', "Hypoxia"),
                (r'\b(cheyne-stokes|cheyne stokes breathing)\b', "Cheyne-Stokes Breathing"),
            ],
            # GI symptoms
            "gastrointestinal": [
                (r'\b(nausea|feeling sick)\b', "Nausea"),
                (r'\b(vomiting|emesis)\b', "Vomiting"),
                (r'\b(constipation|difficulty passing stool)\b', "Constipation"),
                (r'\b(diarrhea|diarrhoea|loose stools)\b', "Diarrhea"),
                (r'\b(bowel obstruction|intestinal obstruction)\b', "Bowel Obstruction"),
                (r'\b(ascites|abdominal fluid)\b', "Ascites"),
                (r'\b(cachexia|wasting|weight loss|muscle wasting)\b', "Cachexia"),
                (r'\b(anorexia|loss of appetite|poor appetite)\b', "Anorexia"),
                (r'\b(dysphagia|difficulty swallowing|swallowing difficulty)\b', "Dysphagia"),
                (r'\b(xerostomia|dry mouth)\b', "Xerostomia"),
                (r'\b(hiccups|hiccoughs)\b', "Hiccups"),
                (r'\b(bloating|abdominal distension)\b', "Bloating"),
            ],
            # Neurological symptoms
            "neurological": [
                (r'\b(delirium|acute confusion|confused state)\b', "Delirium"),
                (r'\b(agitation|restlessness|terminal restlessness)\b', "Agitation"),
                (r'\b(confusion|disorientation)\b', "Confusion"),
                (r'\b(seizures?|convulsions?|fits?)\b', "Seizures"),
                (r'\b(drowsiness|somnolence|sedation)\b', "Drowsiness"),
                (r'\b(insomnia|sleeplessness|difficulty sleeping)\b', "Insomnia"),
                (r'\b(cognitive impairment|memory loss)\b', "Cognitive Impairment"),
                (r'\b(headache|head pain)\b', "Headache"),
                (r'\b(dizziness|vertigo|lightheadedness)\b', "Dizziness"),
            ],
            # Fatigue and weakness
            "fatigue": [
                (r'\b(fatigue|tiredness|exhaustion)\b', "Fatigue"),
                (r'\b(weakness|asthenia|debility)\b', "Weakness"),
                (r'\b(malaise|feeling unwell)\b', "Malaise"),
            ],
            # Skin symptoms
            "skin": [
                (r'\b(pressure ulcer|pressure sore|bedsore|decubitus)\b', "Pressure Ulcer"),
                (r'\b(wound|skin breakdown)\b', "Wound"),
                (r'\b(pruritus|itching|itchy skin)\b', "Pruritus"),
                (r'\b(lymphedema|lymphoedema|swelling)\b', "Lymphedema"),
                (r'\b(edema|oedema|fluid retention)\b', "Edema"),
                (r'\b(jaundice|yellowing)\b', "Jaundice"),
            ],
            # Psychological symptoms
            "psychological": [
                (r'\b(anxiety|anxious|worry|worried)\b', "Anxiety"),
                (r'\b(depression|depressed|low mood)\b', "Depression"),
                (r'\b(distress|psychological distress|emotional distress)\b', "Distress"),
                (r'\b(fear|fearful|scared)\b', "Fear"),
                (r'\b(panic|panic attack)\b', "Panic"),
            ],
            # Other symptoms
            "other": [
                (r'\b(fever|pyrexia|high temperature)\b', "Fever"),
                (r'\b(sweating|diaphoresis|night sweats)\b', "Sweating"),
                (r'\b(bleeding|hemorrhage|haemorrhage)\b', "Bleeding"),
                (r'\b(urinary retention|difficulty urinating)\b', "Urinary Retention"),
                (r'\b(incontinence|urinary incontinence)\b', "Incontinence"),
                (r'\b(thirst|excessive thirst)\b', "Thirst"),
            ],
        }

    def _init_medication_patterns(self):
        """Initialize medication patterns for palliative care."""
        self._medication_patterns = {
            # Strong opioids
            "strong_opioids": [
                (r'\b(morphine|morphine sulphate|morphine sulfate|oramorph|mst)\b', "Morphine"),
                (r'\b(oxycodone|oxynorm|oxycontin)\b', "Oxycodone"),
                (r'\b(fentanyl|duragesic|fentanyl patch)\b', "Fentanyl"),
                (r'\b(hydromorphone|dilaudid)\b', "Hydromorphone"),
                (r'\b(methadone)\b', "Methadone"),
                (r'\b(diamorphine|heroin)\b', "Diamorphine"),
                (r'\b(buprenorphine|butrans|temgesic)\b', "Buprenorphine"),
                (r'\b(alfentanil)\b', "Alfentanil"),
            ],
            # Weak opioids
            "weak_opioids": [
                (r'\b(codeine|codeine phosphate)\b', "Codeine"),
                (r'\b(tramadol|ultram)\b', "Tramadol"),
                (r'\b(dihydrocodeine|df118)\b', "Dihydrocodeine"),
            ],
            # Non-opioid analgesics
            "non_opioid_analgesics": [
                (r'\b(paracetamol|acetaminophen|tylenol)\b', "Paracetamol"),
                (r'\b(ibuprofen|advil|brufen)\b', "Ibuprofen"),
                (r'\b(naproxen|naprosyn)\b', "Naproxen"),
                (r'\b(diclofenac|voltaren)\b', "Diclofenac"),
                (r'\b(aspirin|acetylsalicylic acid)\b', "Aspirin"),
                (r'\b(celecoxib|celebrex)\b', "Celecoxib"),
            ],
            # Adjuvant analgesics
            "adjuvant_analgesics": [
                (r'\b(gabapentin|neurontin)\b', "Gabapentin"),
                (r'\b(pregabalin|lyrica)\b', "Pregabalin"),
                (r'\b(amitriptyline|elavil)\b', "Amitriptyline"),
                (r'\b(duloxetine|cymbalta)\b', "Duloxetine"),
                (r'\b(carbamazepine|tegretol)\b', "Carbamazepine"),
                (r'\b(ketamine)\b', "Ketamine"),
                (r'\b(lidocaine|lignocaine)\b', "Lidocaine"),
            ],
            # Antiemetics
            "antiemetics": [
                (r'\b(ondansetron|zofran)\b', "Ondansetron"),
                (r'\b(metoclopramide|reglan|maxolon)\b', "Metoclopramide"),
                (r'\b(domperidone|motilium)\b', "Domperidone"),
                (r'\b(cyclizine|valoid)\b', "Cyclizine"),
                (r'\b(haloperidol|haldol)\b', "Haloperidol"),
                (r'\b(levomepromazine|nozinan)\b', "Levomepromazine"),
                (r'\b(granisetron|kytril)\b', "Granisetron"),
                (r'\b(prochlorperazine|compazine|stemetil)\b', "Prochlorperazine"),
                (r'\b(dexamethasone|decadron)\b', "Dexamethasone"),
            ],
            # Anxiolytics and sedatives
            "anxiolytics": [
                (r'\b(midazolam|versed)\b', "Midazolam"),
                (r'\b(lorazepam|ativan)\b', "Lorazepam"),
                (r'\b(diazepam|valium)\b', "Diazepam"),
                (r'\b(clonazepam|klonopin)\b', "Clonazepam"),
                (r'\b(phenobarbital|phenobarbitone|luminal)\b', "Phenobarbital"),
            ],
            # Antipsychotics
            "antipsychotics": [
                (r'\b(haloperidol|haldol)\b', "Haloperidol"),
                (r'\b(olanzapine|zyprexa)\b', "Olanzapine"),
                (r'\b(risperidone|risperdal)\b', "Risperidone"),
                (r'\b(quetiapine|seroquel)\b', "Quetiapine"),
            ],
            # Steroids
            "steroids": [
                (r'\b(dexamethasone|decadron)\b', "Dexamethasone"),
                (r'\b(prednisolone|prednisone)\b', "Prednisolone"),
                (r'\b(methylprednisolone|medrol|solu-medrol)\b', "Methylprednisolone"),
                (r'\b(hydrocortisone|cortisol)\b', "Hydrocortisone"),
            ],
            # Laxatives
            "laxatives": [
                (r'\b(lactulose|duphalac)\b', "Lactulose"),
                (r'\b(senna|senokot)\b', "Senna"),
                (r'\b(bisacodyl|dulcolax)\b', "Bisacodyl"),
                (r'\b(docusate|colace)\b', "Docusate"),
                (r'\b(polyethylene glycol|macrogol|movicol|miralax)\b', "Polyethylene Glycol"),
                (r'\b(methylnaltrexone|relistor)\b', "Methylnaltrexone"),
                (r'\b(naloxegol|movantik)\b', "Naloxegol"),
            ],
            # Anticholinergics (for secretions)
            "anticholinergics": [
                (r'\b(hyoscine|scopolamine|buscopan)\b', "Hyoscine"),
                (r'\b(glycopyrronium|glycopyrrolate|robinul)\b', "Glycopyrronium"),
                (r'\b(atropine)\b', "Atropine"),
            ],
            # Other palliative medications
            "other": [
                (r'\b(octreotide|sandostatin)\b', "Octreotide"),
                (r'\b(oxygen|supplemental oxygen|o2)\b', "Oxygen"),
                (r'\b(furosemide|lasix)\b', "Furosemide"),
                (r'\b(spironolactone|aldactone)\b', "Spironolactone"),
                (r'\b(omeprazole|prilosec)\b', "Omeprazole"),
                (r'\b(pantoprazole|protonix)\b', "Pantoprazole"),
                (r'\b(ranitidine|zantac)\b', "Ranitidine"),
            ],
            # Antidepressants
            "antidepressants": [
                (r'\b(sertraline|zoloft)\b', "Sertraline"),
                (r'\b(mirtazapine|remeron)\b', "Mirtazapine"),
                (r'\b(citalopram|celexa)\b', "Citalopram"),
                (r'\b(escitalopram|lexapro)\b', "Escitalopram"),
                (r'\b(fluoxetine|prozac)\b', "Fluoxetine"),
                (r'\b(venlafaxine|effexor)\b', "Venlafaxine"),
            ],
        }

    def _init_condition_patterns(self):
        """Initialize condition patterns for palliative care."""
        self._condition_patterns = {
            # Cancer types
            "cancer": [
                (r'\b(cancer|carcinoma|malignancy|malignant)\b', "Cancer"),
                (r'\b(metastatic cancer|metastases|mets|stage 4|stage iv)\b', "Metastatic Cancer"),
                (r'\b(lung cancer|bronchogenic carcinoma)\b', "Lung Cancer"),
                (r'\b(breast cancer)\b', "Breast Cancer"),
                (r'\b(prostate cancer)\b', "Prostate Cancer"),
                (r'\b(colorectal cancer|colon cancer|bowel cancer)\b', "Colorectal Cancer"),
                (r'\b(pancreatic cancer)\b', "Pancreatic Cancer"),
                (r'\b(liver cancer|hepatocellular carcinoma|hcc)\b', "Liver Cancer"),
                (r'\b(brain tumor|brain tumour|glioblastoma|gbm)\b', "Brain Tumor"),
                (r'\b(leukemia|leukaemia)\b', "Leukemia"),
                (r'\b(lymphoma|hodgkin|non-hodgkin)\b', "Lymphoma"),
            ],
            # Organ failure
            "organ_failure": [
                (r'\b(heart failure|cardiac failure|congestive heart failure|chf)\b', "Heart Failure"),
                (r'\b(end.?stage heart disease|eshd)\b', "End-Stage Heart Disease"),
                (r'\b(kidney failure|renal failure|end.?stage renal disease|esrd|ckd stage 5)\b', "Kidney Failure"),
                (r'\b(liver failure|hepatic failure|end.?stage liver disease|esld|cirrhosis)\b', "Liver Failure"),
                (r'\b(respiratory failure|lung failure)\b', "Respiratory Failure"),
                (r'\b(multi.?organ failure|mof)\b', "Multi-Organ Failure"),
            ],
            # Respiratory diseases
            "respiratory": [
                (r'\b(copd|chronic obstructive pulmonary disease)\b', "COPD"),
                (r'\b(pulmonary fibrosis|ipf|interstitial lung disease)\b', "Pulmonary Fibrosis"),
                (r'\b(emphysema)\b', "Emphysema"),
            ],
            # Neurological diseases
            "neurological": [
                (r'\b(dementia|alzheimer|vascular dementia)\b', "Dementia"),
                (r'\b(parkinson|parkinson\'?s disease)\b', "Parkinson's Disease"),
                (r'\b(motor neuron disease|mnd|als|amyotrophic lateral sclerosis)\b', "Motor Neuron Disease"),
                (r'\b(multiple sclerosis|ms)\b', "Multiple Sclerosis"),
                (r'\b(stroke|cerebrovascular accident|cva)\b', "Stroke"),
                (r'\b(huntington|huntington\'?s disease)\b', "Huntington's Disease"),
            ],
            # Other conditions
            "other": [
                (r'\b(hiv|aids|hiv/aids)\b', "HIV/AIDS"),
                (r'\b(frailty|frail elderly)\b', "Frailty"),
                (r'\b(terminal illness|life.?limiting illness)\b', "Terminal Illness"),
                (r'\b(advanced illness|serious illness)\b', "Advanced Illness"),
            ],
        }

    def _init_care_patterns(self):
        """Initialize care settings and goals patterns."""
        self._care_setting_patterns = [
            (r'\b(hospice|hospice care|inpatient hospice)\b', "Hospice"),
            (r'\b(home care|home-based care|domiciliary care|home hospice)\b', "Home Care"),
            (r'\b(hospital|inpatient|acute care)\b', "Hospital"),
            (r'\b(nursing home|care home|residential care|long.?term care)\b', "Nursing Home"),
            (r'\b(respite care|respite)\b', "Respite Care"),
            (r'\b(outpatient|ambulatory care)\b', "Outpatient"),
            (r'\b(palliative care unit|pcu)\b', "Palliative Care Unit"),
            (r'\b(icu|intensive care|critical care)\b', "ICU"),
        ]

        self._care_goal_patterns = [
            (r'\b(comfort care|comfort measures|comfort.?focused)\b', "Comfort Care"),
            (r'\b(symptom control|symptom management|symptom relief)\b', "Symptom Control"),
            (r'\b(quality of life|qol)\b', "Quality of Life"),
            (r'\b(dignity|maintaining dignity)\b', "Dignity"),
            (r'\b(peaceful death|good death|dying well)\b', "Peaceful Death"),
            (r'\b(pain control|pain management|pain relief)\b', "Pain Control"),
            (r'\b(end.?of.?life care|eol care|eolc)\b', "End-of-Life Care"),
            (r'\b(holistic care|whole.?person care)\b', "Holistic Care"),
            (r'\b(patient.?centered care|person.?centered care)\b', "Patient-Centered Care"),
        ]

    def _init_assessment_patterns(self):
        """Initialize assessment tool patterns."""
        self._assessment_patterns = [
            (r'\b(pain scale|pain score|numeric rating scale|nrs|vas)\b', "Pain Scale"),
            (r'\b(visual analog scale|visual analogue scale)\b', "Visual Analog Scale"),
            (r'\b(ecog|ecog performance status|ecog score)\b', "ECOG Performance Status"),
            (r'\b(karnofsky|karnofsky performance status|kps)\b', "Karnofsky Performance Status"),
            (r'\b(palliative performance scale|pps)\b', "Palliative Performance Scale"),
            (r'\b(edmonton symptom assessment|esas)\b', "Edmonton Symptom Assessment"),
            (r'\b(brief pain inventory|bpi)\b', "Brief Pain Inventory"),
            (r'\b(functional assessment|fast)\b', "Functional Assessment"),
            (r'\b(prognostic score|pap score|ppi)\b', "Prognostic Score"),
            (r'\b(delirium assessment|cam|confusion assessment)\b', "Delirium Assessment"),
            (r'\b(depression screening|phq|gad)\b', "Depression Screening"),
        ]

    def _init_route_patterns(self):
        """Initialize route of administration patterns."""
        self._route_patterns = [
            (r'\b(oral|orally|by mouth|po)\b', "Oral"),
            (r'\b(subcutaneous|sc|subcut|s/c)\b', "Subcutaneous"),
            (r'\b(intravenous|iv|i\.v\.)\b', "Intravenous"),
            (r'\b(intramuscular|im|i\.m\.)\b', "Intramuscular"),
            (r'\b(transdermal|patch|topical)\b', "Transdermal"),
            (r'\b(sublingual|under the tongue|sl)\b', "Sublingual"),
            (r'\b(buccal)\b', "Buccal"),
            (r'\b(rectal|pr|per rectum)\b', "Rectal"),
            (r'\b(nebulized|nebuliser|nebulizer)\b', "Nebulized"),
            (r'\b(epidural|intrathecal|spinal)\b', "Epidural/Intrathecal"),
            (r'\b(pca|patient.?controlled analgesia)\b', "PCA"),
            (r'\b(syringe driver|syringe pump|csci)\b', "Syringe Driver"),
        ]

    def _init_psychosocial_patterns(self):
        """Initialize psychosocial and spiritual patterns."""
        self._psychosocial_patterns = [
            (r'\b(grief|grieving|mourning)\b', "Grief"),
            (r'\b(bereavement|loss|bereaved)\b', "Bereavement"),
            (r'\b(caregiver burden|carer stress|family stress)\b', "Caregiver Burden"),
            (r'\b(spiritual distress|spiritual pain|existential distress)\b', "Spiritual Distress"),
            (r'\b(social isolation|loneliness|lonely)\b', "Social Isolation"),
            (r'\b(family support|family care|family meeting)\b', "Family Support"),
            (r'\b(counseling|counselling|therapy|psychotherapy)\b', "Counseling"),
            (r'\b(chaplain|spiritual care|pastoral care)\b', "Spiritual Care"),
            (r'\b(social worker|social work)\b', "Social Work"),
            (r'\b(legacy|life review|meaning.?making)\b', "Legacy Work"),
        ]

        self._advance_care_patterns = [
            (r'\b(dnr|dnar|do not resuscitate|do not attempt resuscitation)\b', "DNR"),
            (r'\b(advance directive|living will|healthcare directive)\b', "Advance Directive"),
            (r'\b(goals of care|goc|care goals discussion)\b', "Goals of Care"),
            (r'\b(polst|molst|physician orders for life.?sustaining treatment)\b', "POLST"),
            (r'\b(healthcare proxy|health care agent|surrogate decision.?maker)\b', "Healthcare Proxy"),
            (r'\b(prognosis discussion|prognostic awareness)\b', "Prognosis Discussion"),
            (r'\b(serious illness conversation|advance care planning|acp)\b', "Advance Care Planning"),
            (r'\b(code status|resuscitation status|full code|comfort measures only|cmo)\b', "Code Status"),
            (r'\b(withdrawal of treatment|withholding treatment)\b', "Withdrawal of Treatment"),
        ]

    def _init_intervention_patterns(self):
        """Initialize intervention and procedure patterns."""
        self._intervention_patterns = [
            # Medical procedures
            (r'\b(paracentesis|abdominal tap|ascitic tap)\b', "Paracentesis"),
            (r'\b(thoracentesis|pleural tap|chest tap)\b', "Thoracentesis"),
            (r'\b(nerve block|neurolytic block|celiac plexus block)\b', "Nerve Block"),
            (r'\b(palliative sedation|terminal sedation|continuous sedation)\b', "Palliative Sedation"),
            (r'\b(blood transfusion|transfusion)\b', "Blood Transfusion"),
            (r'\b(wound care|wound dressing|pressure ulcer care)\b', "Wound Care"),
            (r'\b(tracheostomy care|trach care)\b', "Tracheostomy Care"),
            (r'\b(feeding tube|peg tube|ng tube|enteral feeding)\b', "Feeding Tube"),
            (r'\b(catheter care|urinary catheter|foley)\b', "Catheter Care"),
            (r'\b(stent|biliary stent|ureteral stent)\b', "Stent Placement"),
            # Non-pharmacological interventions
            (r'\b(massage|massage therapy)\b', "Massage Therapy"),
            (r'\b(music therapy)\b', "Music Therapy"),
            (r'\b(aromatherapy)\b', "Aromatherapy"),
            (r'\b(relaxation techniques?|breathing exercises?)\b', "Relaxation Techniques"),
            (r'\b(physiotherapy|physical therapy)\b', "Physiotherapy"),
            (r'\b(occupational therapy)\b', "Occupational Therapy"),
            (r'\b(speech therapy|speech and language)\b', "Speech Therapy"),
            (r'\b(repositioning|turning|pressure relief)\b', "Repositioning"),
            (r'\b(mouth care|oral care|oral hygiene)\b', "Mouth Care"),
            (r'\b(hydration|artificial hydration|iv fluids)\b', "Hydration"),
        ]

    async def extract(
        self,
        text: str,
        use_llm: bool = True
    ) -> Tuple[List[Entity], List[Relationship]]:
        """
        Extract entities and relationships from text.

        Args:
            text: Text to analyze
            use_llm: Whether to use LLM extraction

        Returns:
            Tuple of (entities, relationships)
        """
        entities = []
        relationships = []

        # Pattern-based extraction (fast, always available)
        if self.use_patterns:
            pattern_entities = self._extract_with_patterns(text)
            entities.extend(pattern_entities)

        # LLM-based extraction (more accurate, requires API)
        if use_llm and (self._groq_client or self._openai_client):
            try:
                llm_entities, llm_relationships = await self._extract_with_llm(text)
                entities.extend(llm_entities)
                relationships.extend(llm_relationships)
            except Exception as e:
                logger.error(f"LLM extraction failed: {e}")

        # Deduplicate entities
        entities = self._deduplicate_entities(entities)

        return entities, relationships

    def _extract_with_patterns(self, text: str) -> List[Entity]:
        """Extract entities using regex patterns."""
        entities = []
        text_lower = text.lower()

        # Extract symptoms
        for category, patterns in self._symptom_patterns.items():
            for pattern, name in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    entities.append(Entity(
                        name=name,
                        type=EntityType.SYMPTOM,
                        properties={"category": category},
                        source_text=text[:100],
                        confidence=0.85
                    ))

        # Extract medications
        for category, patterns in self._medication_patterns.items():
            for pattern, name in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    entities.append(Entity(
                        name=name,
                        type=EntityType.MEDICATION,
                        properties={"category": category},
                        source_text=text[:100],
                        confidence=0.9
                    ))

        # Extract conditions
        for category, patterns in self._condition_patterns.items():
            for pattern, name in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    entities.append(Entity(
                        name=name,
                        type=EntityType.CONDITION,
                        properties={"category": category},
                        source_text=text[:100],
                        confidence=0.85
                    ))

        # Extract care settings
        for pattern, name in self._care_setting_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                entities.append(Entity(
                    name=name,
                    type=EntityType.CARE_SETTING,
                    source_text=text[:100],
                    confidence=0.9
                ))

        # Extract care goals
        for pattern, name in self._care_goal_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                entities.append(Entity(
                    name=name,
                    type=EntityType.CARE_GOAL,
                    source_text=text[:100],
                    confidence=0.85
                ))

        # Extract assessment tools
        for pattern, name in self._assessment_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                entities.append(Entity(
                    name=name,
                    type=EntityType.ASSESSMENT_TOOL,
                    source_text=text[:100],
                    confidence=0.9
                ))

        # Extract routes
        for pattern, name in self._route_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                entities.append(Entity(
                    name=name,
                    type=EntityType.ROUTE,
                    source_text=text[:100],
                    confidence=0.9
                ))

        # Extract psychosocial
        for pattern, name in self._psychosocial_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                entities.append(Entity(
                    name=name,
                    type=EntityType.PSYCHOSOCIAL,
                    source_text=text[:100],
                    confidence=0.8
                ))

        # Extract advance care planning
        for pattern, name in self._advance_care_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                entities.append(Entity(
                    name=name,
                    type=EntityType.ADVANCE_CARE,
                    source_text=text[:100],
                    confidence=0.9
                ))

        # Extract interventions
        for pattern, name in self._intervention_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                entities.append(Entity(
                    name=name,
                    type=EntityType.INTERVENTION,
                    source_text=text[:100],
                    confidence=0.85
                ))

        return entities

    async def _extract_with_llm(
        self,
        text: str
    ) -> Tuple[List[Entity], List[Relationship]]:
        """Extract entities using LLM."""
        prompt = ENTITY_EXTRACTION_PROMPT.format(text=text[:4000])

        try:
            if self._groq_client:
                response = await self._groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=2000,
                )
                content = response.choices[0].message.content

            elif self._openai_client:
                response = await self._openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=2000,
                )
                content = response.choices[0].message.content

            else:
                return [], []

            return self._parse_llm_response(content)

        except Exception as e:
            logger.error(f"LLM extraction error: {e}")
            return [], []

    def _parse_llm_response(
        self,
        content: str
    ) -> Tuple[List[Entity], List[Relationship]]:
        """Parse LLM JSON response into entities and relationships."""
        entities = []
        relationships = []

        try:
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', content)
            if not json_match:
                return [], []

            data = json.loads(json_match.group())

            # Parse entities
            for e in data.get("entities", []):
                try:
                    entity_type = EntityType(e.get("type", "Symptom"))
                    entities.append(Entity(
                        name=e.get("name", ""),
                        type=entity_type,
                        properties=e.get("properties", {}),
                        confidence=0.95
                    ))
                except ValueError:
                    continue

            # Parse relationships
            for r in data.get("relationships", []):
                try:
                    rel_type = RelationshipType(r.get("relationship", "TREATS"))
                    source_type = EntityType(r.get("source_type", "Medication"))
                    target_type = EntityType(r.get("target_type", "Symptom"))

                    relationships.append(Relationship(
                        source=r.get("source", ""),
                        source_type=source_type,
                        target=r.get("target", ""),
                        target_type=target_type,
                        relationship=rel_type,
                        properties=r.get("properties", {}),
                        confidence=0.9
                    ))
                except ValueError:
                    continue

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")

        return entities, relationships

    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities, keeping highest confidence."""
        seen = {}
        for entity in entities:
            key = (entity.name.lower(), entity.type)
            if key not in seen or entity.confidence > seen[key].confidence:
                seen[key] = entity

        return list(seen.values())

    def extract_from_chunks(
        self,
        chunks: List[str]
    ) -> Tuple[List[Entity], List[Relationship]]:
        """
        Synchronously extract entities from multiple text chunks.
        Uses pattern matching only (no LLM).

        Args:
            chunks: List of text chunks

        Returns:
            Tuple of (entities, relationships)
        """
        all_entities = []
        for chunk in chunks:
            entities = self._extract_with_patterns(chunk)
            all_entities.extend(entities)

        return self._deduplicate_entities(all_entities), []
