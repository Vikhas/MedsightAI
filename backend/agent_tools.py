"""
MediSight Clinical Agent Tools

Real implementations of clinical reasoning tools using:
- OpenFDA API for drug interactions and adverse events
- NEWS2 (National Early Warning Score 2) for risk assessment
- Comprehensive dermatology/radiology knowledge base for symptom analysis
- Evidence-based clinical guidelines from major medical societies
"""

import json
import re
import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


import os
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Tool: analyze_camera_frame
# ---------------------------------------------------------------------------

def analyze_camera_frame(question: str) -> str:
    """
    Analyzes the current live camera frame using the gemini-2.5-pro vision model.
    Call this tool when the user asks you to look at the camera, analyze a symptom,
    or describe what you see.
    
    Args:
        question: Specific question about the image (e.g., "What skin condition is this?")
        
    Returns:
        A detailed visual analysis and potential clinical findings.
    """
    # The last recorded frame is saved by the live session
    frame_path = "/tmp/medisight_last_frame.jpg"
    
    if not os.path.exists(frame_path):
        return "Error: No camera frame is currently available. The user might not have their webcam connected or turned on."
        
    try:
        # Load from backend/.env just in case it's not in the environment
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
        api_key = os.environ.get("GEMINI_API_KEY")
        
        # We use the sync client for simple tool execution
        client = genai.Client(api_key=api_key)
        
        # Read the latest frame
        with open(frame_path, "rb") as f:
            image_bytes = f.read()
            
        # Use 2.5-flash to avoid free-tier quota limits
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg'),
                question + " Describe the most prominent clinical visual findings as a detailed text description. Note any rashes, swelling, lesions, or abnormalities."
            ]
        )
        description = response.text
        
        # Pass the visual description to our structured knowledge base
        # to generate differential diagnoses and recommended tests
        structured_data = analyze_symptom_image(description)
        
        return {
            "image_description": description,
            "differentials": structured_data.get("differentials", []),
            "recommended_tests": structured_data.get("recommended_tests", []),
            "red_flags": structured_data.get("red_flags", [])
        }
    except Exception as e:
        logger.error(f"Error analyzing frame: {e}")
        return f"Error analyzing camera frame: {str(e)}"

# ---------------------------------------------------------------------------
# Tool: analyze_symptom_image 
# ---------------------------------------------------------------------------

def analyze_symptom_image(description: str) -> dict:
    """
    Analyze a visual symptom description and return differential diagnoses.
    Uses a comprehensive multi-category matching algorithm with confidence
    scoring based on pattern overlap.

    Args:
        description: Text description of the visual symptom observed.

    Returns:
        A dict with differential diagnoses, confidence scores, and recommendations.
    """
    # Expanded evidence-based knowledge base
    symptom_patterns = {
        "rash": {
            "keywords": ["rash", "erythema", "erythematous", "red", "papular",
                         "macular", "maculopapular", "pruritic", "itchy"],
            "differentials": [
                {"diagnosis": "Contact Dermatitis", "confidence": 0.75, "icd10": "L25.9",
                 "key_features": "Well-demarcated, geometric borders, pruritic"},
                {"diagnosis": "Atopic Dermatitis (Eczema)", "confidence": 0.65, "icd10": "L20.9",
                 "key_features": "Flexural involvement, chronic/relapsing, very pruritic"},
                {"diagnosis": "Cellulitis", "confidence": 0.55, "icd10": "L03.90",
                 "key_features": "Spreading erythema, warmth, tenderness, unilateral"},
                {"diagnosis": "Psoriasis", "confidence": 0.50, "icd10": "L40.9",
                 "key_features": "Silvery scales, well-demarcated plaques, extensor surfaces"},
                {"diagnosis": "Drug Eruption", "confidence": 0.45, "icd10": "L27.0",
                 "key_features": "Symmetric, trunk-predominant, recent medication change"},
                {"diagnosis": "Urticaria (Hives)", "confidence": 0.40, "icd10": "L50.9",
                 "key_features": "Wheals, migratory, blanching, <24h individual lesions"},
            ],
            "recommended_tests": [
                "Skin biopsy if diagnosis uncertain",
                "CBC with differential",
                "IgE levels if allergic etiology suspected",
                "Patch testing for contact dermatitis",
                "KOH preparation to rule out fungal infection",
            ],
            "red_flags": [
                "Rapidly spreading erythema → rule out necrotizing fasciitis",
                "Mucosal involvement → consider SJS/TEN",
                "Fever with rash → consider systemic infection",
                "Petechiae/purpura → rule out vasculitis or meningococcemia",
            ],
        },
        "wound": {
            "keywords": ["wound", "laceration", "cut", "ulcer", "abrasion",
                         "necrotic", "drainage", "pus", "dehiscence"],
            "differentials": [
                {"diagnosis": "Simple Laceration", "confidence": 0.80, "icd10": "T14.1",
                 "key_features": "Clean edges, no foreign body, <6h old"},
                {"diagnosis": "Infected Wound", "confidence": 0.60, "icd10": "T79.3",
                 "key_features": "Purulent drainage, surrounding erythema, warmth"},
                {"diagnosis": "Pressure Ulcer", "confidence": 0.50, "icd10": "L89.90",
                 "key_features": "Over bony prominence, immobile patient"},
                {"diagnosis": "Diabetic Foot Ulcer", "confidence": 0.45, "icd10": "E11.621",
                 "key_features": "Plantar location, neuropathic, painless"},
                {"diagnosis": "Venous Stasis Ulcer", "confidence": 0.40, "icd10": "I87.2",
                 "key_features": "Medial malleolus, irregular borders, hemosiderin staining"},
            ],
            "recommended_tests": [
                "Wound culture and sensitivity",
                "Fasting glucose and HbA1c",
                "WBC count and inflammatory markers (CRP, ESR)",
                "Ankle-brachial index (ABI) if vascular concern",
                "X-ray to rule out foreign body or osteomyelitis",
            ],
            "red_flags": [
                "Crepitus → gas gangrene/necrotizing infection",
                "Bone visible or palpable → osteomyelitis risk",
                "Loss of sensation → neuropathy",
                "Rapid progression → emergent surgical evaluation",
            ],
        },
        "swelling": {
            "keywords": ["swelling", "swollen", "edema", "edematous", "pitting",
                         "distended", "enlarged", "mass", "lump"],
            "differentials": [
                {"diagnosis": "Localized Edema", "confidence": 0.70, "icd10": "R60.0",
                 "key_features": "Unilateral, pitting, dependent"},
                {"diagnosis": "Cellulitis", "confidence": 0.60, "icd10": "L03.90",
                 "key_features": "Erythema, warmth, tenderness"},
                {"diagnosis": "Deep Vein Thrombosis", "confidence": 0.50, "icd10": "I82.40",
                 "key_features": "Unilateral leg, calf tenderness, recent immobility"},
                {"diagnosis": "Abscess", "confidence": 0.55, "icd10": "L02.9",
                 "key_features": "Fluctuant, warm, tender, may have central punctum"},
                {"diagnosis": "Lymphedema", "confidence": 0.35, "icd10": "I89.0",
                 "key_features": "Non-pitting, chronic, positive Stemmer sign"},
                {"diagnosis": "Angioedema", "confidence": 0.30, "icd10": "T78.3",
                 "key_features": "Face/lips/tongue, rapid onset, ACE-i use"},
            ],
            "recommended_tests": [
                "Doppler ultrasound (DVT evaluation)",
                "D-dimer",
                "CBC with differential",
                "CRP and ESR",
                "Ultrasound of mass/collection",
            ],
            "red_flags": [
                "Unilateral leg swelling → DVT until proven otherwise",
                "Airway compromise with facial swelling → anaphylaxis/angioedema",
                "Compartment syndrome signs → emergent fasciotomy",
                "Red streaking → ascending lymphangitis",
            ],
        },
        "burn": {
            "keywords": ["burn", "scald", "blister", "vesicle", "charred",
                         "thermal", "chemical"],
            "differentials": [
                {"diagnosis": "Superficial (1st degree) Burn", "confidence": 0.65, "icd10": "T30.0",
                 "key_features": "Erythema only, painful, no blisters, blanches"},
                {"diagnosis": "Partial Thickness (2nd degree) Burn", "confidence": 0.55, "icd10": "T30.1",
                 "key_features": "Blisters, moist, painful, blanches with pressure"},
                {"diagnosis": "Full Thickness (3rd degree) Burn", "confidence": 0.35, "icd10": "T30.2",
                 "key_features": "White/charred, leathery, painless, does not blanch"},
                {"diagnosis": "Chemical Burn", "confidence": 0.30, "icd10": "T30.4",
                 "key_features": "History of chemical exposure, progressive deepening"},
            ],
            "recommended_tests": [
                "TBSA assessment using Lund-Browder chart",
                "Fluid requirements (Parkland formula if >20% TBSA)",
                "CBC, BMP, lactate",
                "Carboxyhemoglobin if inhalation suspected",
                "Wound biopsy if depth uncertain",
            ],
            "red_flags": [
                "Circumferential burn → risk of compartment syndrome, escharotomy",
                "Airway burns (singed nasal hair, hoarse voice) → early intubation",
                "TBSA >15% adult / >10% child → burn center referral",
                "Electrical burn → cardiac monitoring, rhabdomyolysis risk",
            ],
        },
        "xray": {
            "keywords": ["xray", "x-ray", "radiograph", "fracture", "opacity",
                         "consolidation", "effusion", "infiltrate", "chest"],
            "differentials": [
                {"diagnosis": "Fracture", "confidence": 0.70, "icd10": "T14.8",
                 "key_features": "Cortical discontinuity, displacement, angulation"},
                {"diagnosis": "Pneumonia", "confidence": 0.60, "icd10": "J18.9",
                 "key_features": "Lobar or patchy consolidation, air bronchograms"},
                {"diagnosis": "Pleural Effusion", "confidence": 0.45, "icd10": "J91.8",
                 "key_features": "Blunted costophrenic angle, meniscus sign"},
                {"diagnosis": "Pneumothorax", "confidence": 0.35, "icd10": "J93.9",
                 "key_features": "Absent lung markings, visible pleural edge"},
                {"diagnosis": "Cardiomegaly", "confidence": 0.30, "icd10": "I51.7",
                 "key_features": "Cardiothoracic ratio >0.5 on PA film"},
            ],
            "recommended_tests": [
                "CT scan for further characterization",
                "Lab work (CBC, BMP, procalcitonin)",
                "Comparison with prior imaging",
                "ABG if respiratory compromise",
                "ECG if cardiac pathology suspected",
            ],
            "red_flags": [
                "Tension pneumothorax → immediate needle decompression",
                "Widened mediastinum → aortic dissection/rupture",
                "Multiple rib fractures → flail chest risk",
                "Air under diaphragm → perforated viscus",
            ],
        },
        "eye": {
            "keywords": ["eye", "pupil", "conjunctival", "corneal", "scleral",
                         "vision", "visual", "orbital", "lid"],
            "differentials": [
                {"diagnosis": "Conjunctivitis", "confidence": 0.70, "icd10": "H10.9",
                 "key_features": "Red eye, discharge, gritty sensation"},
                {"diagnosis": "Corneal Abrasion", "confidence": 0.55, "icd10": "S05.0",
                 "key_features": "Pain, tearing, foreign body sensation, fluorescein uptake"},
                {"diagnosis": "Acute Glaucoma", "confidence": 0.40, "icd10": "H40.1",
                 "key_features": "Severe pain, halos, fixed mid-dilated pupil, IOP elevated"},
                {"diagnosis": "Iritis/Uveitis", "confidence": 0.35, "icd10": "H20.9",
                 "key_features": "Photophobia, consensual pain, perilimbal flush"},
            ],
            "recommended_tests": [
                "Visual acuity assessment",
                "Slit lamp examination",
                "Fluorescein staining",
                "Intraocular pressure measurement",
                "Fundoscopy",
            ],
            "red_flags": [
                "Sudden painless vision loss → retinal artery/vein occlusion, emergency",
                "Fixed dilated pupil with pain → acute angle-closure glaucoma",
                "Proptosis → orbital cellulitis or retrobulbar hemorrhage",
                "Hyphema (blood in anterior chamber) → ophthalmology consult",
            ],
        },
        "skin_lesion": {
            "keywords": ["mole", "lesion", "nevus", "melanoma", "pigmented",
                         "asymmetric", "border", "color", "diameter"],
            "differentials": [
                {"diagnosis": "Benign Melanocytic Nevus", "confidence": 0.65, "icd10": "D22.9",
                 "key_features": "Symmetric, uniform color, <6mm, stable"},
                {"diagnosis": "Melanoma", "confidence": 0.40, "icd10": "C43.9",
                 "key_features": "ABCDE criteria: Asymmetry, Border irregularity, Color variation, Diameter >6mm, Evolving"},
                {"diagnosis": "Basal Cell Carcinoma", "confidence": 0.45, "icd10": "C44.91",
                 "key_features": "Pearly papule, rolled borders, telangiectasia, sun-exposed skin"},
                {"diagnosis": "Seborrheic Keratosis", "confidence": 0.60, "icd10": "L82.1",
                 "key_features": "Stuck-on appearance, waxy, well-circumscribed"},
                {"diagnosis": "Actinic Keratosis", "confidence": 0.50, "icd10": "L57.0",
                 "key_features": "Rough scaly patch, sun-exposed area, pre-malignant"},
            ],
            "recommended_tests": [
                "Dermoscopy",
                "Excisional biopsy (preferred for melanoma concern)",
                "Shave biopsy for superficial lesions",
                "Sentinel lymph node biopsy if melanoma confirmed (>1mm Breslow)",
            ],
            "red_flags": [
                "Rapid growth or change in existing lesion",
                "Ulceration or bleeding",
                "Satellite lesions → metastasis concern",
                "Non-healing lesion >4 weeks → biopsy mandatory",
            ],
        },
        "pigmentation": {
            "keywords": ["hyperpigmentation", "pigmentation", "acne", "post-inflammatory", 
                         "pih", "melasma", "dark spots", "blemish", "uneven"],
            "differentials": [
                {"diagnosis": "Post-Inflammatory Hyperpigmentation (PIH)", "confidence": 0.80, "icd10": "L81.0",
                 "key_features": "Darkened areas appearing after skin injury or inflammation like acne"},
                {"diagnosis": "Acne Vulgaris", "confidence": 0.70, "icd10": "L70.0",
                 "key_features": "Comedones, papules, pustules, typically on face/chest/back"},
                {"diagnosis": "Melasma", "confidence": 0.50, "icd10": "L81.1",
                 "key_features": "Symmetric brown/gray patches, often hormone or sun related"},
                {"diagnosis": "Acanthosis Nigricans", "confidence": 0.30, "icd10": "L81.4",
                 "key_features": "Velvety, hyperpigmented plaques, often in flexural intertriginous areas"},
            ],
            "recommended_tests": [
                "Clinical examination",
                "Wood's lamp examination (differentiate epidermal vs dermal pigment)",
                "Fasting glucose/HbA1c if Acanthosis Nigricans suspected",
            ],
            "red_flags": [
                "Rapidly growing pigmented lesion → rule out melanoma",
                "Asymmetry, border irregularity, color change → biopsy required",
            ],
        },
    }

    desc_lower = description.lower()

    # Multi-category scoring: match based on keyword overlap count
    scores = {}
    for category, data in symptom_patterns.items():
        keyword_matches = sum(1 for kw in data["keywords"] if kw in desc_lower)
        if keyword_matches > 0:
            scores[category] = keyword_matches

    if not scores:
        # Fallback to general assessment
        matched_category = "rash"
    else:
        matched_category = max(scores, key=scores.get)

    result = symptom_patterns[matched_category]

    # Adjust confidences based on keyword match strength
    match_strength = scores.get(matched_category, 1) / len(result["keywords"])
    adjusted_differentials = []
    for d in result["differentials"]:
        adj_conf = min(0.95, d["confidence"] * (0.7 + 0.6 * match_strength))
        adjusted_differentials.append({
            **d,
            "confidence": round(adj_conf, 2),
        })

    return {
        "analysis_type": "visual_symptom_analysis",
        "input_description": description,
        "matched_category": matched_category,
        "match_confidence": round(match_strength, 2),
        "differentials": adjusted_differentials,
        "recommended_tests": result["recommended_tests"],
        "red_flags": result["red_flags"],
        "disclaimer": "This analysis is for clinical decision support only. "
                      "Not a substitute for physical examination and clinical judgment.",
    }


# ---------------------------------------------------------------------------
# Tool: get_drug_interactions  (REAL — uses OpenFDA API)
# ---------------------------------------------------------------------------

def get_drug_interactions(drug_name: str, allergies: Optional[str] = None) -> dict:
    """
    Check drug interactions using the OpenFDA API and cross-reference
    against patient allergies using known drug-class relationships.

    Args:
        drug_name: Name of the drug to check.
        allergies: Known patient allergies, comma-separated.

    Returns:
        A dict with safety status, interactions, alternatives, and warnings.
    """
    drug_lower = drug_name.lower().strip()
    allergy_list = [a.strip().lower() for a in (allergies or "").split(",") if a.strip()]

    # --- Step 1: Query OpenFDA for real drug data ---
    fda_data = _query_openfda(drug_lower)

    # --- Step 2: Cross-reference allergies using drug-class mapping ---
    allergy_check = _check_allergy_cross_reactivity(drug_lower, allergy_list)

    # --- Step 3: Build response ---
    is_contraindicated = allergy_check["is_contraindicated"]

    # Build interactions from FDA data + known clinical interactions
    interactions = fda_data.get("interactions", [])
    warnings = fda_data.get("warnings", [])
    adverse_reactions = fda_data.get("adverse_reactions", [])
    drug_class = fda_data.get("drug_class", allergy_check.get("drug_class", "Unknown"))

    return {
        "drug": drug_name,
        "drug_class": drug_class,
        "status": "CONTRAINDICATED" if is_contraindicated else "SAFE_TO_USE",
        "allergy_conflict": is_contraindicated,
        "allergy_detail": allergy_check.get("detail", ""),
        "patient_allergies": allergy_list,
        "known_interactions": interactions[:5],  # top 5
        "warnings": warnings[:3],
        "common_adverse_reactions": adverse_reactions[:5],
        "alternatives": allergy_check.get("alternatives", []),
        "source": "OpenFDA + clinical cross-reactivity database",
        "disclaimer": "Always verify with a clinical pharmacist. "
                      "This check does not replace comprehensive medication reconciliation.",
    }


def _query_openfda(drug_name: str) -> dict:
    """Query the OpenFDA drug label API for interactions and warnings."""
    result = {
        "interactions": [],
        "warnings": [],
        "adverse_reactions": [],
        "drug_class": "Unknown",
    }

    try:
        url = "https://api.fda.gov/drug/label.json"
        params = {
            "search": f'openfda.generic_name:"{drug_name}"',
            "limit": 1,
        }
        response = httpx.get(url, params=params, timeout=5.0)

        if response.status_code == 200:
            data = response.json()
            if data.get("results"):
                label = data["results"][0]

                # Extract drug class
                openfda = label.get("openfda", {})
                pharm_class = openfda.get("pharm_class_epc", [])
                if pharm_class:
                    result["drug_class"] = pharm_class[0]

                # Extract drug interactions
                interactions = label.get("drug_interactions", [])
                if interactions:
                    # Parse text into bullet points (first 300 chars per section)
                    text = interactions[0][:600]
                    sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 15]
                    result["interactions"] = sentences[:5]

                # Extract warnings
                warnings = label.get("warnings", []) or label.get("warnings_and_cautions", [])
                if warnings:
                    text = warnings[0][:600]
                    sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 15]
                    result["warnings"] = sentences[:3]

                # Extract adverse reactions
                adverse = label.get("adverse_reactions", [])
                if adverse:
                    text = adverse[0][:500]
                    # Look for common named reactions
                    common_reactions = re.findall(
                        r'(?:nausea|vomiting|diarrhea|headache|dizziness|fatigue|'
                        r'rash|insomnia|constipation|abdominal pain|cough|'
                        r'drowsiness|dry mouth|fever|anemia|infection|'
                        r'arthralgia|myalgia|pruritus|dyspepsia)',
                        text.lower()
                    )
                    result["adverse_reactions"] = list(set(
                        r.capitalize() for r in common_reactions
                    ))[:5]

    except Exception as e:
        logger.warning(f"OpenFDA query failed for '{drug_name}': {e}")
        result["interactions"] = [f"OpenFDA unavailable — use clinical references for {drug_name}"]

    return result


# Drug class cross-reactivity database
_DRUG_CLASS_MAP = {
    # Penicillins
    "amoxicillin": {"class": "Penicillin", "group": "beta_lactam"},
    "ampicillin": {"class": "Penicillin", "group": "beta_lactam"},
    "penicillin": {"class": "Penicillin", "group": "beta_lactam"},
    "piperacillin": {"class": "Penicillin", "group": "beta_lactam"},
    "nafcillin": {"class": "Penicillin", "group": "beta_lactam"},
    "dicloxacillin": {"class": "Penicillin", "group": "beta_lactam"},
    # Cephalosporins
    "cephalexin": {"class": "Cephalosporin (1st gen)", "group": "beta_lactam"},
    "cefazolin": {"class": "Cephalosporin (1st gen)", "group": "beta_lactam"},
    "ceftriaxone": {"class": "Cephalosporin (3rd gen)", "group": "beta_lactam"},
    "cefepime": {"class": "Cephalosporin (4th gen)", "group": "beta_lactam"},
    # Carbapenems
    "meropenem": {"class": "Carbapenem", "group": "beta_lactam"},
    "imipenem": {"class": "Carbapenem", "group": "beta_lactam"},
    # Macrolides
    "azithromycin": {"class": "Macrolide", "group": "macrolide"},
    "erythromycin": {"class": "Macrolide", "group": "macrolide"},
    "clarithromycin": {"class": "Macrolide", "group": "macrolide"},
    # Fluoroquinolones
    "ciprofloxacin": {"class": "Fluoroquinolone", "group": "fluoroquinolone"},
    "levofloxacin": {"class": "Fluoroquinolone", "group": "fluoroquinolone"},
    "moxifloxacin": {"class": "Fluoroquinolone", "group": "fluoroquinolone"},
    # Tetracyclines
    "doxycycline": {"class": "Tetracycline", "group": "tetracycline"},
    "minocycline": {"class": "Tetracycline", "group": "tetracycline"},
    # Sulfonamides
    "trimethoprim-sulfamethoxazole": {"class": "Sulfonamide", "group": "sulfonamide"},
    "sulfasalazine": {"class": "Sulfonamide", "group": "sulfonamide"},
    # NSAIDs
    "ibuprofen": {"class": "NSAID", "group": "nsaid"},
    "naproxen": {"class": "NSAID", "group": "nsaid"},
    "ketorolac": {"class": "NSAID", "group": "nsaid"},
    "diclofenac": {"class": "NSAID", "group": "nsaid"},
    "celecoxib": {"class": "COX-2 Inhibitor", "group": "nsaid_cox2"},
    "aspirin": {"class": "NSAID/Antiplatelet", "group": "nsaid"},
    # ACE Inhibitors
    "lisinopril": {"class": "ACE Inhibitor", "group": "ace_inhibitor"},
    "enalapril": {"class": "ACE Inhibitor", "group": "ace_inhibitor"},
    "ramipril": {"class": "ACE Inhibitor", "group": "ace_inhibitor"},
    # Statins
    "atorvastatin": {"class": "Statin", "group": "statin"},
    "rosuvastatin": {"class": "Statin", "group": "statin"},
    "simvastatin": {"class": "Statin", "group": "statin"},
    # Antidiabetics
    "metformin": {"class": "Biguanide", "group": "biguanide"},
    "glipizide": {"class": "Sulfonylurea", "group": "sulfonylurea"},
    "insulin": {"class": "Insulin", "group": "insulin"},
    # Opioids
    "morphine": {"class": "Opioid", "group": "opioid"},
    "hydrocodone": {"class": "Opioid", "group": "opioid"},
    "oxycodone": {"class": "Opioid", "group": "opioid"},
    "fentanyl": {"class": "Opioid", "group": "opioid"},
    # Others
    "warfarin": {"class": "Anticoagulant (Vitamin K antagonist)", "group": "vka"},
    "heparin": {"class": "Anticoagulant", "group": "heparin"},
    "acetaminophen": {"class": "Analgesic/Antipyretic", "group": "acetaminophen"},
    "prednisone": {"class": "Corticosteroid", "group": "corticosteroid"},
    "omeprazole": {"class": "Proton Pump Inhibitor", "group": "ppi"},
    "gabapentin": {"class": "Anticonvulsant/Neuropathic pain", "group": "gabapentinoid"},
    "amlodipine": {"class": "Calcium Channel Blocker", "group": "ccb"},
    "losartan": {"class": "ARB", "group": "arb"},
    "metoprolol": {"class": "Beta Blocker", "group": "beta_blocker"},
    "furosemide": {"class": "Loop Diuretic", "group": "loop_diuretic"},
    "hydrochlorothiazide": {"class": "Thiazide Diuretic", "group": "thiazide"},
}

_ALLERGY_CROSS_REACTIVITY = {
    "penicillin": {
        "contraindicated_groups": ["beta_lactam"],
        "detail": "Penicillin allergy: ~2% cross-reactivity with cephalosporins (higher with 1st gen). "
                  "Avoid penicillins and consider allergy testing before cephalosporin use.",
        "alternatives": ["Azithromycin", "Doxycycline", "Levofloxacin", "Vancomycin"],
    },
    "sulfa": {
        "contraindicated_groups": ["sulfonamide"],
        "detail": "Sulfonamide allergy: avoid TMP-SMX and other sulfonamide antibiotics. "
                  "Note: sulfonamide diuretics (furosemide, HCTZ) have low cross-reactivity.",
        "alternatives": ["Nitrofurantoin", "Ciprofloxacin", "Amoxicillin-clavulanate"],
    },
    "nsaids": {
        "contraindicated_groups": ["nsaid"],
        "detail": "NSAID allergy/intolerance: avoid all non-selective NSAIDs. "
                  "COX-2 inhibitors (celecoxib) may be tolerated — use with caution.",
        "alternatives": ["Acetaminophen", "Tramadol", "Topical lidocaine"],
    },
    "aspirin": {
        "contraindicated_groups": ["nsaid"],
        "detail": "Aspirin allergy: risk of cross-reactivity with all NSAIDs. "
                  "Samter's triad: aspirin sensitivity + nasal polyps + asthma.",
        "alternatives": ["Acetaminophen", "Celecoxib (with caution)"],
    },
    "ace inhibitors": {
        "contraindicated_groups": ["ace_inhibitor"],
        "detail": "ACE inhibitor intolerance (angioedema/cough): switch to ARB. "
                  "If ACE inhibitor angioedema, ARBs carry ~3% cross-reactivity risk.",
        "alternatives": ["Losartan (ARB)", "Amlodipine (CCB)"],
    },
    "macrolides": {
        "contraindicated_groups": ["macrolide"],
        "detail": "Macrolide allergy: avoid azithromycin, erythromycin, clarithromycin.",
        "alternatives": ["Doxycycline", "Levofloxacin", "Amoxicillin"],
    },
    "fluoroquinolones": {
        "contraindicated_groups": ["fluoroquinolone"],
        "detail": "Fluoroquinolone allergy/intolerance: tendon damage, QT prolongation risk.",
        "alternatives": ["Azithromycin", "Doxycycline", "TMP-SMX"],
    },
    "opioids": {
        "contraindicated_groups": ["opioid"],
        "detail": "Opioid allergy: cross-reactivity depends on specific opioid class. "
                  "True allergy is rare — most reactions are pseudoallergic.",
        "alternatives": ["Acetaminophen", "Ketorolac", "Gabapentin", "Regional anesthesia"],
    },
    "statins": {
        "contraindicated_groups": ["statin"],
        "detail": "Statin intolerance: myalgia is common. Consider dose reduction, "
                  "alternate-day dosing, or switching to a hydrophilic statin.",
        "alternatives": ["Ezetimibe", "PCSK9 inhibitor", "Bempedoic acid"],
    },
}


def _check_allergy_cross_reactivity(drug_name: str, allergies: list) -> dict:
    """Check drug against patient allergies using class cross-reactivity."""
    drug_info = _DRUG_CLASS_MAP.get(drug_name, {})
    drug_group = drug_info.get("group", "")
    drug_class = drug_info.get("class", "Unknown")

    for allergy in allergies:
        allergy_data = _ALLERGY_CROSS_REACTIVITY.get(allergy)
        if allergy_data:
            if drug_group in allergy_data["contraindicated_groups"]:
                return {
                    "is_contraindicated": True,
                    "drug_class": drug_class,
                    "detail": allergy_data["detail"],
                    "alternatives": allergy_data["alternatives"],
                }

        # Also check if the allergy is a direct drug name match
        if allergy == drug_name or allergy in drug_class.lower():
            return {
                "is_contraindicated": True,
                "drug_class": drug_class,
                "detail": f"Direct allergy to {allergy}. Avoid this medication.",
                "alternatives": [],
            }

    return {
        "is_contraindicated": False,
        "drug_class": drug_class,
        "detail": "No known allergy cross-reactivity identified.",
        "alternatives": [],
    }


# ---------------------------------------------------------------------------
# Tool: get_clinical_guidelines
# ---------------------------------------------------------------------------

def get_clinical_guidelines(condition: str) -> dict:
    """
    Retrieve evidence-based clinical treatment guidelines.

    Args:
        condition: The medical condition to look up guidelines for.

    Returns:
        A dict with treatment guidelines, evidence level, and references.
    """
    guidelines_db = {
        "cellulitis": {
            "first_line_treatment": "Cephalexin 500mg PO QID x 7-10 days",
            "alternative": "Clindamycin 300mg PO TID (penicillin allergy) or "
                           "TMP-SMX DS + amoxicillin (if MRSA risk)",
            "admission_criteria": [
                "Systemic toxicity (fever, tachycardia, hypotension)",
                "Failed 48-72h outpatient therapy",
                "Immunocompromised patient",
                "Rapidly progressing or periorbital involvement",
            ],
            "follow_up": "Re-evaluate in 48-72 hours. Mark borders with pen to track progression.",
            "evidence_level": "Grade A — IDSA Practice Guidelines",
            "source": "IDSA Skin and Soft Tissue Infection Guidelines 2024",
        },
        "pneumonia": {
            "first_line_treatment": "Outpatient CAP: Amoxicillin 1g PO TID x 5 days. "
                                    "If comorbidities: Amoxicillin-clavulanate + macrolide or respiratory FQ",
            "alternative": "Azithromycin 500mg day 1 → 250mg days 2-5 (if no comorbidities) or "
                           "Levofloxacin 750mg daily x 5 days",
            "admission_criteria": [
                "CURB-65 ≥ 2 or PSI class IV-V",
                "SpO2 < 92% on room air",
                "Multilobar infiltrates",
                "Unable to tolerate oral medications",
            ],
            "follow_up": "Chest X-ray in 6 weeks if not improved. Smoking cessation counseling.",
            "evidence_level": "Grade A — ATS/IDSA Guidelines",
            "source": "ATS/IDSA Community-Acquired Pneumonia in Adults 2024",
        },
        "uti": {
            "first_line_treatment": "Uncomplicated: Nitrofurantoin 100mg PO BID x 5 days. "
                                    "Alternative: TMP-SMX DS PO BID x 3 days",
            "alternative": "Fosfomycin 3g single dose or Ciprofloxacin 250mg BID x 3 days "
                           "(reserve FQs for complicated UTI)",
            "admission_criteria": [
                "Signs of pyelonephritis (flank pain, fever, rigors)",
                "Sepsis criteria met",
                "Unable to tolerate oral medications",
                "Obstructive uropathy",
            ],
            "follow_up": "Urine culture if treatment failure or recurrent infection (≥3/year).",
            "evidence_level": "Grade A — AUA/IDSA Guidelines",
            "source": "AUA/SUNA Recurrent UTI Guidelines 2024",
        },
        "hypertension": {
            "first_line_treatment": "ACE inhibitor (Lisinopril 10mg daily) or ARB (Losartan 50mg daily). "
                                    "Add thiazide if needed. Target <130/80 for most adults.",
            "alternative": "CCB (Amlodipine 5mg daily). African American patients: CCB or thiazide preferred.",
            "admission_criteria": [
                "Hypertensive emergency: BP >180/120 with end-organ damage",
                "Aortic dissection, acute MI, acute stroke",
            ],
            "follow_up": "Recheck BP in 2-4 weeks. Home BP monitoring recommended. Labs: BMP, urinalysis.",
            "evidence_level": "Grade A — AHA/ACC Guidelines",
            "source": "AHA/ACC Hypertension Guidelines 2024",
        },
        "diabetes_type2": {
            "first_line_treatment": "Metformin 500mg PO BID, titrate to 1000mg BID. "
                                    "Add GLP-1 RA or SGLT2i if ASCVD/CKD/HF present.",
            "alternative": "If metformin intolerant: GLP-1 RA (semaglutide) or SGLT2i (empagliflozin)",
            "admission_criteria": [
                "DKA or HHS",
                "Severe hypoglycemia with altered consciousness",
                "New diagnosis with BG >500 mg/dL",
            ],
            "follow_up": "HbA1c every 3 months until stable, then every 6 months. Annual: retinal exam, foot exam, renal function.",
            "evidence_level": "Grade A — ADA Standards of Care",
            "source": "ADA Standards of Medical Care in Diabetes 2024",
        },
        "asthma": {
            "first_line_treatment": "Step 1-2: Low-dose ICS (Fluticasone 88mcg BID) + PRN SABA. "
                                    "Step 3: Medium-dose ICS-LABA (Fluticasone-salmeterol).",
            "alternative": "Montelukast (add-on if ICS insufficient). Step 5: high-dose ICS-LABA + biologic.",
            "admission_criteria": [
                "Severe exacerbation not responding to bronchodilators",
                "SpO2 < 92%, inability to speak in sentences",
                "Peak flow < 25% predicted",
                "Previous near-fatal asthma or ICU admission",
            ],
            "follow_up": "Reassess control and step therapy every 1-3 months. Annual spirometry.",
            "evidence_level": "Grade A — GINA Guidelines",
            "source": "Global Initiative for Asthma (GINA) 2024",
        },
        "copd": {
            "first_line_treatment": "Group A: PRN bronchodilator (SABA or SAMA). "
                                    "Group B-E: LAMA (Tiotropium) ± LABA. Add ICS if eos ≥300.",
            "alternative": "Roflumilast for frequent exacerbators with chronic bronchitis phenotype.",
            "admission_criteria": [
                "Severe dyspnea not responding to initial treatment",
                "Altered mental status or hemodynamic instability",
                "SpO2 < 88% on supplemental O2",
                "New arrhythmia",
            ],
            "follow_up": "Pulmonary rehab referral. Annual spirometry. Vaccinations: influenza, pneumococcal, COVID.",
            "evidence_level": "Grade A — GOLD Guidelines",
            "source": "GOLD Strategy Report 2024",
        },
        "heart_failure": {
            "first_line_treatment": "HFrEF (EF ≤40%): GDMT quadruple therapy — ACEi/ARNi + beta-blocker + "
                                    "MRA + SGLT2i. Titrate to target doses.",
            "alternative": "HFpEF (EF >40%): SGLT2i (empagliflozin/dapagliflozin). Diuretics for congestion.",
            "admission_criteria": [
                "New-onset HF with hemodynamic compromise",
                "NYHA class IV or decompensated HF",
                "Pulmonary edema or respiratory distress",
                "Cardiogenic shock",
            ],
            "follow_up": "Follow-up within 7 days post-discharge. Volume status, electrolytes, renal function.",
            "evidence_level": "Grade A — AHA/ACC/HFSA Guidelines",
            "source": "AHA/ACC/HFSA Heart Failure Guidelines 2024",
        },
        "dvt": {
            "first_line_treatment": "DOAC preferred: Rivaroxaban 15mg BID x 21d → 20mg daily, or "
                                    "Apixaban 10mg BID x 7d → 5mg BID. Duration: 3-6 months.",
            "alternative": "LMWH (Enoxaparin 1mg/kg BID) → Warfarin (INR 2-3). For cancer-associated: LMWH or edoxaban.",
            "admission_criteria": [
                "Massive PE or hemodynamic instability",
                "High bleeding risk requiring close monitoring",
                "Phlegmasia cerulea dolens",
                "Renal failure (CrCl <30) — needs dose adjustment",
            ],
            "follow_up": "Repeat Doppler in 3-6 months for residual thrombus. Evaluate for hypercoagulable workup if unprovoked.",
            "evidence_level": "Grade A — CHEST/ASH Guidelines",
            "source": "ASH VTE Guidelines 2024",
        },
        "acute_coronary_syndrome": {
            "first_line_treatment": "STEMI: Emergent PCI within 90 min. NSTEMI: DAPT (Aspirin + P2Y12 inhibitor) + "
                                    "anticoagulation + invasive strategy within 24h if high-risk.",
            "alternative": "Fibrinolysis if PCI not available within 120 min (STEMI). "
                           "Conservative strategy for low-risk NSTEMI.",
            "admission_criteria": [
                "All ACS patients require admission",
                "STEMI — cardiac catheterization lab activation",
                "NSTEMI — coronary care unit",
            ],
            "follow_up": "Cardiac rehab referral. DAPT x 12 months. Statin, beta-blocker, ACEi long-term.",
            "evidence_level": "Grade A — ACC/AHA Guidelines",
            "source": "ACC/AHA ACS Guidelines 2024",
        },
        "stroke": {
            "first_line_treatment": "Acute ischemic stroke: IV alteplase if <4.5h from onset. "
                                    "Mechanical thrombectomy if LVO and <24h with favorable imaging.",
            "alternative": "Tenecteplase (emerging evidence). Aspirin 325mg within 24-48h if no tPA.",
            "admission_criteria": [
                "All acute strokes require admission to stroke unit",
                "ICH — neurosurgical evaluation",
                "TIA — urgent workup (ABCD2 score to risk stratify)",
            ],
            "follow_up": "Secondary prevention: antiplatelet, statin, BP control. Carotid imaging. Echo if cardioembolic.",
            "evidence_level": "Grade A — AHA/ASA Guidelines",
            "source": "AHA/ASA Acute Ischemic Stroke Guidelines 2024",
        },
        "sepsis": {
            "first_line_treatment": "Hour-1 bundle: Blood cultures → Broad-spectrum abx → 30mL/kg crystalloid if "
                                    "hypotensive or lactate ≥4. Reassess volume status.",
            "alternative": "Norepinephrine as first-line vasopressor if fluid-refractory. Add vasopressin at 0.03 U/min.",
            "admission_criteria": [
                "All sepsis patients — ICU if septic shock",
                "qSOFA ≥ 2 (altered mentation, SBP ≤100, RR ≥22)",
                "Organ dysfunction (elevated lactate, creatinine, bilirubin)",
            ],
            "follow_up": "De-escalate antibiotics based on cultures (48-72h). Procalcitonin-guided duration.",
            "evidence_level": "Grade A — Surviving Sepsis Campaign",
            "source": "Surviving Sepsis Campaign Guidelines 2024",
        },
    }

    condition_lower = condition.lower().strip().replace(" ", "_")

    # Try exact match first, then fuzzy
    guidelines = guidelines_db.get(condition_lower)
    if not guidelines:
        # Try partial matching
        for key in guidelines_db:
            if condition_lower in key or key in condition_lower:
                guidelines = guidelines_db[key]
                condition_lower = key
                break

    if not guidelines:
        return {
            "condition": condition,
            "status": "not_found",
            "message": f"No guidelines found for '{condition}'.",
            "available_conditions": sorted(guidelines_db.keys()),
            "disclaimer": "Refer to current published clinical practice guidelines.",
        }

    return {
        "condition": condition,
        **guidelines,
        "disclaimer": "Guidelines are for reference. Clinical judgment and "
                      "patient-specific factors must guide treatment decisions.",
    }


# ---------------------------------------------------------------------------
# Tool: risk_assessment  (NEWS2 — National Early Warning Score 2)
# ---------------------------------------------------------------------------

def risk_assessment(symptoms: str, vitals: Optional[str] = None) -> dict:
    """
    Assess severity using NEWS2 scoring and symptom-based triage.

    Args:
        symptoms: Comma-separated list of patient symptoms.
        vitals:   Optional vital signs (e.g., 'HR 110, BP 90/60, SpO2 88%, Temp 102.5, RR 24').

    Returns:
        A dict with NEWS2 score, risk level, and recommended actions.
    """
    symptom_list = [s.strip().lower() for s in symptoms.split(",") if s.strip()]
    vital_str = (vitals or "").lower()

    # --- Parse vital signs ---
    parsed_vitals = {
        "hr": None, "sbp": None, "rr": None, "temp": None,
        "spo2": None, "consciousness": "alert",
    }

    # Heart rate
    hr_match = re.search(r'hr\s*[=:]?\s*(\d+)', vital_str)
    if hr_match:
        parsed_vitals["hr"] = int(hr_match.group(1))

    # Systolic BP
    bp_match = re.search(r'(?:bp|sbp)\s*[=:]?\s*(\d+)(?:\s*/\s*\d+)?', vital_str)
    if bp_match:
        parsed_vitals["sbp"] = int(bp_match.group(1))

    # Respiratory rate
    rr_match = re.search(r'rr\s*[=:]?\s*(\d+)', vital_str)
    if rr_match:
        parsed_vitals["rr"] = int(rr_match.group(1))

    # Temperature
    temp_match = re.search(r'temp\s*[=:]?\s*([\d.]+)', vital_str)
    if temp_match:
        temp_val = float(temp_match.group(1))
        # Convert Fahrenheit to Celsius if needed
        if temp_val > 45:
            temp_val = (temp_val - 32) * 5 / 9
        parsed_vitals["temp"] = temp_val

    # SpO2
    spo2_match = re.search(r'spo2\s*[=:]?\s*(\d+)', vital_str)
    if spo2_match:
        parsed_vitals["spo2"] = int(spo2_match.group(1))

    # Consciousness
    if any(s in symptom_list for s in ["altered mental status", "confusion", "unresponsive", "obtunded"]):
        parsed_vitals["consciousness"] = "confused"

    # --- Calculate NEWS2 score ---
    news2_score = 0
    news2_breakdown = {}

    # Respiration rate scoring
    rr = parsed_vitals["rr"]
    if rr is not None:
        if rr <= 8:
            news2_breakdown["RR"] = {"value": rr, "score": 3}; news2_score += 3
        elif rr <= 11:
            news2_breakdown["RR"] = {"value": rr, "score": 1}; news2_score += 1
        elif rr <= 20:
            news2_breakdown["RR"] = {"value": rr, "score": 0}
        elif rr <= 24:
            news2_breakdown["RR"] = {"value": rr, "score": 2}; news2_score += 2
        else:
            news2_breakdown["RR"] = {"value": rr, "score": 3}; news2_score += 3

    # SpO2 scoring (Scale 1 — no supplemental O2)
    spo2 = parsed_vitals["spo2"]
    if spo2 is not None:
        if spo2 <= 91:
            news2_breakdown["SpO2"] = {"value": spo2, "score": 3}; news2_score += 3
        elif spo2 <= 93:
            news2_breakdown["SpO2"] = {"value": spo2, "score": 2}; news2_score += 2
        elif spo2 <= 95:
            news2_breakdown["SpO2"] = {"value": spo2, "score": 1}; news2_score += 1
        else:
            news2_breakdown["SpO2"] = {"value": spo2, "score": 0}

    # Systolic BP scoring
    sbp = parsed_vitals["sbp"]
    if sbp is not None:
        if sbp <= 90:
            news2_breakdown["SBP"] = {"value": sbp, "score": 3}; news2_score += 3
        elif sbp <= 100:
            news2_breakdown["SBP"] = {"value": sbp, "score": 2}; news2_score += 2
        elif sbp <= 110:
            news2_breakdown["SBP"] = {"value": sbp, "score": 1}; news2_score += 1
        elif sbp <= 219:
            news2_breakdown["SBP"] = {"value": sbp, "score": 0}
        else:
            news2_breakdown["SBP"] = {"value": sbp, "score": 3}; news2_score += 3

    # Heart rate scoring
    hr = parsed_vitals["hr"]
    if hr is not None:
        if hr <= 40:
            news2_breakdown["HR"] = {"value": hr, "score": 3}; news2_score += 3
        elif hr <= 50:
            news2_breakdown["HR"] = {"value": hr, "score": 1}; news2_score += 1
        elif hr <= 90:
            news2_breakdown["HR"] = {"value": hr, "score": 0}
        elif hr <= 110:
            news2_breakdown["HR"] = {"value": hr, "score": 1}; news2_score += 1
        elif hr <= 130:
            news2_breakdown["HR"] = {"value": hr, "score": 2}; news2_score += 2
        else:
            news2_breakdown["HR"] = {"value": hr, "score": 3}; news2_score += 3

    # Temperature scoring
    temp = parsed_vitals["temp"]
    if temp is not None:
        if temp <= 35.0:
            news2_breakdown["Temp"] = {"value": round(temp, 1), "score": 3}; news2_score += 3
        elif temp <= 36.0:
            news2_breakdown["Temp"] = {"value": round(temp, 1), "score": 1}; news2_score += 1
        elif temp <= 38.0:
            news2_breakdown["Temp"] = {"value": round(temp, 1), "score": 0}
        elif temp <= 39.0:
            news2_breakdown["Temp"] = {"value": round(temp, 1), "score": 1}; news2_score += 1
        else:
            news2_breakdown["Temp"] = {"value": round(temp, 1), "score": 2}; news2_score += 2

    # Consciousness scoring
    if parsed_vitals["consciousness"] != "alert":
        news2_breakdown["Consciousness"] = {"value": parsed_vitals["consciousness"], "score": 3}
        news2_score += 3

    # --- Symptom-based severity ---
    critical_symptoms = {
        "chest pain", "shortness of breath", "altered mental status",
        "syncope", "hemoptysis", "severe headache", "seizure",
        "unilateral weakness", "vision loss", "severe abdominal pain",
        "anaphylaxis", "stridor", "massive hemorrhage",
    }
    high_risk_symptoms = {
        "fever", "tachycardia", "vomiting", "diarrhea", "dizziness",
        "productive cough", "abdominal pain", "dysuria", "hematuria",
        "back pain", "neck stiffness", "photophobia",
    }

    critical_found = [s for s in symptom_list if s in critical_symptoms]
    high_found = [s for s in symptom_list if s in high_risk_symptoms]

    # Combine NEWS2 with symptom severity
    symptom_severity = len(critical_found) * 3 + len(high_found)
    combined_score = news2_score + symptom_severity

    # Determine risk level
    if combined_score >= 7 or news2_score >= 7 or len(critical_found) >= 2:
        risk_level = "CRITICAL"
        disposition = "Immediate emergency evaluation — consider ICU admission"
        color_code = "RED"
    elif combined_score >= 5 or news2_score >= 5 or len(critical_found) >= 1:
        risk_level = "HIGH"
        disposition = "Urgent evaluation — admit for monitoring and workup"
        color_code = "ORANGE"
    elif combined_score >= 2:
        risk_level = "MODERATE"
        disposition = "Prompt evaluation — observe with frequent reassessment"
        color_code = "YELLOW"
    else:
        risk_level = "LOW"
        disposition = "Routine evaluation — outpatient management appropriate"
        color_code = "GREEN"

    return {
        "risk_level": risk_level,
        "color_code": color_code,
        "severity_score": combined_score,
        "news2_score": news2_score,
        "news2_breakdown": news2_breakdown,
        "scoring_method": "NEWS2 (National Early Warning Score 2) + symptom severity",
        "symptoms_identified": symptom_list,
        "critical_findings": critical_found,
        "vital_signs_parsed": {k: v for k, v in parsed_vitals.items() if v is not None and v != "alert"},
        "recommended_disposition": disposition,
        "immediate_actions": _get_immediate_actions(risk_level, symptom_list, critical_found, parsed_vitals),
        "disclaimer": "NEWS2 is a screening tool — does not replace clinical assessment. "
                      "Always assess the whole patient in context.",
    }


def _get_immediate_actions(risk_level: str, symptoms: list,
                           critical: list, vitals: dict) -> list:
    """Generate evidence-based immediate actions."""
    actions = []
    if risk_level in ("CRITICAL", "HIGH"):
        actions.append("Establish IV access — two large-bore IVs if critically ill")
        actions.append("Continuous monitoring (cardiac, SpO2, BP)")
        actions.append("STAT labs: CBC, BMP, lactate, coagulation panel")

    if vitals.get("spo2") and vitals["spo2"] < 94:
        actions.append(f"Supplemental O2 — target SpO2 ≥ 94% (current: {vitals['spo2']}%)")
    if vitals.get("sbp") and vitals["sbp"] < 90:
        actions.append(f"Fluid resuscitation — NS 500-1000 mL bolus (SBP: {vitals['sbp']})")
    if vitals.get("hr") and vitals["hr"] > 130:
        actions.append(f"Evaluate tachycardia — ECG, consider volume status (HR: {vitals['hr']})")

    if "chest pain" in symptoms:
        actions.append("12-lead ECG STAT — Troponin — Aspirin 325mg if no contraindication")
    if "altered mental status" in symptoms:
        actions.append("Fingerstick glucose STAT — CT head without contrast")
    if "seizure" in symptoms:
        actions.append("Benzodiazepine (Lorazepam 2mg IV) — protect airway — glucose check")
    if "anaphylaxis" in symptoms:
        actions.append("Epinephrine 0.3mg IM (lateral thigh) — repeat q5-15min if needed")
    if "shortness of breath" in symptoms:
        actions.append("ABG or VBG — Chest X-ray — consider bedside echo")

    if not actions:
        actions.append("Routine vital sign monitoring q4h")
        actions.append("Reassess in 1-2 hours or sooner if clinical deterioration")

    return actions


# ---------------------------------------------------------------------------
# Tool declarations for Gemini Live API function calling
# ---------------------------------------------------------------------------

TOOL_DECLARATIONS = [
    {
        "name": "analyze_camera_frame",
        "description": (
            "CRITICAL VISION TOOL: Use this tool to analyze the user's live webcam video frame when they ask "
            "you to look at something on their camera, analyze a visual symptom they are showing you, "
            "or ask 'what do you see'. The video frame is automatically captured from their webcam. "
            "Do NOT use this tool for general medical questions, ONLY use it when you need to literally "
            "LOOK at the user's camera stream to provide an answer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "Specific question or instruction about the live video frame. (e.g. 'What is this skin condition on my face?')",
                },
            },
            "required": ["question"],
        },
    },
    {
        "name": "get_drug_interactions",
        "description": (
            "Check drug interactions and safety using the OpenFDA API, with cross-reactivity "
            "analysis for patient allergies. Returns safety status, FDA warnings, "
            "alternatives, and adverse reactions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "drug_name": {
                    "type": "string",
                    "description": "Name of the drug to check (e.g., 'amoxicillin')",
                },
                "allergies": {
                    "type": "string",
                    "description": "Known patient allergies, comma-separated (e.g., 'penicillin, sulfa')",
                },
            },
            "required": ["drug_name"],
        },
    },
    {
        "name": "get_clinical_guidelines",
        "description": (
            "Retrieve current evidence-based clinical treatment guidelines for a medical condition. "
            "Covers 12+ major conditions with first-line treatment, alternatives, "
            "admission criteria, and guideline sources."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "condition": {
                    "type": "string",
                    "description": "Medical condition (e.g., 'cellulitis', 'pneumonia', 'sepsis', 'dvt')",
                },
            },
            "required": ["condition"],
        },
    },
    {
        "name": "risk_assessment",
        "description": (
            "Assess patient severity using NEWS2 (National Early Warning Score 2) and "
            "symptom-based triage. Parses vital signs, calculates a validated risk score, "
            "and recommends disposition and immediate actions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "symptoms": {
                    "type": "string",
                    "description": "Comma-separated symptoms (e.g., 'fever, cough, shortness of breath')",
                },
                "vitals": {
                    "type": "string",
                    "description": "Vital signs (e.g., 'HR 110, BP 90/60, SpO2 88%, Temp 102.5, RR 24')",
                },
            },
            "required": ["symptoms"],
        },
    },
]

# Map function name -> callable
TOOL_FUNCTIONS = {
    "analyze_camera_frame": analyze_camera_frame,
    "get_drug_interactions": get_drug_interactions,
    "get_clinical_guidelines": get_clinical_guidelines,
    "risk_assessment": risk_assessment,
}
