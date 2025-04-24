#!/usr/bin/env python3
"""
Medical Diagnosis Example using Hyperlogica

This example demonstrates how to use the Hyperlogica system for medical diagnosis
support, analyzing patient symptoms and medical data to suggest potential diagnoses
with confidence levels and recommended next steps.

Usage:
    python medical_diagnosis_example.py [--config CONFIG_FILE] [--patients PATIENTS_FILE] 
                                       [--output OUTPUT_FILE] [--verbose]

The example will:
1. Load medical knowledge base and diagnostic rules
2. Process patient records with symptoms and test results
3. Apply Bayesian reasoning to generate diagnostic suggestions
4. Output detailed analysis with explanations and recommendations
"""

import os
import json
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

# Import the main hyperlogica module
from hyperlogica import process_input_file


# Sample medical diagnosis configuration
SAMPLE_MEDICAL_CONFIG = {
    "processing": {
        "vector_dimension": 10000,
        "vector_type": "continuous",  # Use continuous vectors for medical concepts
        "reasoning_approach": "bayesian",  # Use Bayesian approach for medical diagnosis
        "certainty_propagation": "noisy_or",  # Multiple symptom patterns can indicate the same diagnosis
        "recalibration_enabled": True,
        "max_reasoning_depth": 6,
        "domain": "medical",
        "domain_config": {
            "prior_positive": 0.2,  # Prior probability for positive diagnosis
            "prior_negative": 0.7,  # Prior probability for negative diagnosis (healthy)
            "prior_neutral": 0.1,   # Prior probability for indeterminate
            "positive_outcome_keywords": ["confirmed", "probable", "likely", "consistent with"],
            "negative_outcome_keywords": ["ruled out", "unlikely", "inconsistent with", "negative for"],
            "neutral_outcome_keywords": ["possible", "indeterminate", "requires further testing", "differential"],
            "outcome_field": "diagnostic_assessment",
            "positive_outcome": "PROBABLE_DIAGNOSIS",
            "negative_outcome": "UNLIKELY_DIAGNOSIS",
            "neutral_outcome": "POSSIBLE_DIAGNOSIS"
        }
    },
    "persistence": {
        "load_previous_state": False,
        "save_state": True,
        "state_save_path": "./output/medical_diagnosis_state.pkl"
    },
    "logging": {
        "log_level": "info",
        "log_path": "./logs/medical_diagnosis.log",
        "include_vector_operations": False,
        "include_llm_interactions": True,
        "include_reasoning_steps": True
    },
    "llm": {
        "model": "gpt-4",
        "temperature": 0.0,
        "max_tokens": 2000
    },
    "input_data": {
        "rules": [
            {"text": "If a patient presents with fever, cough, and shortness of breath, then they may have pneumonia.", "certainty": 0.7},
            {"text": "If a patient has elevated white blood cell count and pneumonia symptoms, then bacterial pneumonia is likely.", "certainty": 0.8},
            {"text": "If a patient has normal white blood cell count and pneumonia symptoms, then viral pneumonia is more likely.", "certainty": 0.75},
            {"text": "If a chest X-ray shows consolidation, then pneumonia is highly probable.", "certainty": 0.9},
            {"text": "If a patient has fever, rash, and joint pain, then they may have an autoimmune condition.", "certainty": 0.6},
            {"text": "If a patient has positive ANA test and joint symptoms, then lupus should be considered.", "certainty": 0.8},
            {"text": "If a patient has chest pain, shortness of breath, and elevated troponin, then acute coronary syndrome is likely.", "certainty": 0.85},
            {"text": "If an ECG shows ST elevation and the patient has chest pain, then STEMI is probable.", "certainty": 0.9},
            {"text": "If a patient has severe headache of sudden onset with nuchal rigidity, then meningitis is possible.", "certainty": 0.7},
            {"text": "If CSF analysis shows elevated WBC and protein with decreased glucose, then bacterial meningitis is likely.", "certainty": 0.9},
            {"text": "If a patient has unintended weight loss, fatigue, and night sweats, then malignancy should be considered.", "certainty": 0.6},
            {"text": "If a patient is over 60 with memory loss and cognitive decline, then dementia is possible.", "certainty": 0.7},
            {"text": "If a patient has polyuria, polydipsia, and weight loss, then diabetes mellitus is likely.", "certainty": 0.8},
            {"text": "If blood glucose is consistently above 126 mg/dL fasting, then diabetes mellitus is confirmed.", "certainty": 0.95},
            {"text": "If a patient has intermittent fever, headache, and joint pain with recent travel to tropical regions, then consider malaria.", "certainty": 0.7}
        ],
        "entities": [
            {
                "id": "patient_A",
                "name": "Patient A",
                "demographics": {
                    "age": 45,
                    "sex": "male",
                    "risk_factors": ["smoker", "hypertension"]
                },
                "facts": [
                    {"text": "Patient presents with fever of 101.5F for 3 days.", "certainty": 0.95},
                    {"text": "Patient reports productive cough with yellow sputum.", "certainty": 0.9},
                    {"text": "Patient reports shortness of breath on exertion.", "certainty": 0.9},
                    {"text": "Physical exam reveals crackles in right lower lobe.", "certainty": 0.9},
                    {"text": "White blood cell count is elevated at 14,500/μL.", "certainty": 0.95},
                    {"text": "Chest X-ray shows consolidation in right lower lobe.", "certainty": 0.95},
                    {"text": "Oxygen saturation is 94% on room air.", "certainty": 0.95},
                    {"text": "Patient has no recent travel history.", "certainty": 0.9}
                ]
            },
            {
                "id": "patient_B",
                "name": "Patient B",
                "demographics": {
                    "age": 62,
                    "sex": "female",
                    "risk_factors": ["diabetes", "obesity"]
                },
                "facts": [
                    {"text": "Patient presents with chest pain described as pressure, 8/10 intensity.", "certainty": 0.95},
                    {"text": "Pain radiates to left arm and jaw.", "certainty": 0.9},
                    {"text": "Patient reports shortness of breath and nausea.", "certainty": 0.9},
                    {"text": "ECG shows ST elevation in leads II, III, and aVF.", "certainty": 0.95},
                    {"text": "Troponin I is elevated at 2.3 ng/mL.", "certainty": 0.95},
                    {"text": "Patient has history of hyperlipidemia.", "certainty": 0.9},
                    {"text": "Blood pressure is 160/95 mmHg.", "certainty": 0.95},
                    {"text": "Patient reports similar but milder episode last week.", "certainty": 0.8}
                ]
            },
            {
                "id": "patient_C",
                "name": "Patient C",
                "demographics": {
                    "age": 28,
                    "sex": "female",
                    "risk_factors": ["recent travel to Southeast Asia"]
                },
                "facts": [
                    {"text": "Patient presents with intermittent fever for 5 days.", "certainty": 0.95},
                    {"text": "Fever pattern shows spikes every 48 hours.", "certainty": 0.9},
                    {"text": "Patient reports severe headache and muscle pain.", "certainty": 0.9},
                    {"text": "Patient returned from Thailand 2 weeks ago.", "certainty": 0.95},
                    {"text": "Physical exam shows mild splenomegaly.", "certainty": 0.8},
                    {"text": "Complete blood count shows mild thrombocytopenia.", "certainty": 0.9},
                    {"text": "Rapid diagnostic test for malaria is positive.", "certainty": 0.95},
                    {"text": "Patient reports no prophylactic antimalarial use during travel.", "certainty": 0.9}
                ]
            }
        ]
    },
    "output_schema": {
        "format": "json",
        "fields": [
            {"name": "patient_id", "type": "string"},
            {"name": "patient_name", "type": "string"},
            {"name": "demographics", "type": "object"},
            {"name": "diagnostic_assessment", "type": "string"},
            {"name": "certainty", "type": "float"},
            {"name": "differential_diagnoses", "type": "array"},
            {"name": "recommended_tests", "type": "array"},
            {"name": "reasoning", "type": "object"}
        ],
        "include_reasoning_trace": True,
        "include_explanation": True,
        "include_vector_details": False,
        "domain": "medical"
    }
}


def create_or_load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a new configuration or load from file if provided.
    
    Args:
        config_path (str, optional): Path to existing configuration file
        
    Returns:
        dict: Configuration dictionary for medical diagnosis
    """
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    
    return SAMPLE_MEDICAL_CONFIG


def load_patients_from_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Load patient records from a JSON file.
    
    Args:
        file_path (str): Path to JSON file containing patient records
        
    Returns:
        list: List of patient records as entities for Hyperlogica
    """
    if not os.path.exists(file_path):
        logging.warning(f"Patient file not found: {file_path}")
        return []
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Validate expected structure
        if not isinstance(data, list):
            logging.error("Patient file must contain a list of patient records")
            return []
        
        patients = []
        for i, patient in enumerate(data):
            # Check for required fields
            if "symptoms" not in patient:
                logging.warning(f"Patient record {i} missing symptoms, skipping")
                continue
            
            # Build facts from symptoms and test results
            facts = []
            
            # Process symptoms
            for symptom in patient.get("symptoms", []):
                facts.append({
                    "text": symptom,
                    "certainty": 0.9  # High certainty for reported symptoms
                })
            
            # Process test results
            for test in patient.get("test_results", []):
                if "name" in test and "result" in test:
                    facts.append({
                        "text": f"{test['name']} result is {test['result']}.",
                        "certainty": 0.95  # Very high certainty for test results
                    })
            
            # Process medical history
            for condition in patient.get("medical_history", []):
                facts.append({
                    "text": f"Patient has history of {condition}.",
                    "certainty": 0.9
                })
            
            # Create entity for this patient
            patient_entity = {
                "id": patient.get("id", f"patient_{i}"),
                "name": patient.get("name", f"Patient {i}"),
                "demographics": patient.get("demographics", {}),
                "facts": facts
            }
            
            patients.append(patient_entity)
            logging.info(f"Loaded patient: {patient_entity['name']} with {len(facts)} facts")
        
        return patients
        
    except Exception as e:
        logging.error(f"Error loading patients from {file_path}: {str(e)}")
        return []


def add_differential_diagnoses(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add differential diagnoses based on reasoning results.
    
    Args:
        results (dict): Results from hyperlogica processing
        
    Returns:
        dict: Updated results with differential diagnoses
    """
    # Process each patient result
    for result in results.get("results", []):
        reasoning = result.get("reasoning", {})
        diagnosis = result.get("outcome", "UNKNOWN")
        
        # Extract all mentioned conditions from conclusions
        conditions = []
        for factor in reasoning.get("key_factors", []):
            # Extract condition name from factor
            factor_id = factor.get("factor", "")
            certainty = factor.get("certainty", 0)
            
            # Simple extraction - in real implementation would use more sophisticated methods
            conditions_mentioned = []
            
            if "pneumonia" in factor_id.lower():
                conditions_mentioned.append(("Pneumonia", certainty))
            if "bacterial" in factor_id.lower():
                conditions_mentioned.append(("Bacterial Infection", certainty))
            if "viral" in factor_id.lower():
                conditions_mentioned.append(("Viral Infection", certainty))
            if "coronary" in factor_id.lower() or "stemi" in factor_id.lower():
                conditions_mentioned.append(("Acute Coronary Syndrome", certainty))
            if "malaria" in factor_id.lower():
                conditions_mentioned.append(("Malaria", certainty))
            if "diabetes" in factor_id.lower():
                conditions_mentioned.append(("Diabetes Mellitus", certainty))
            if "meningitis" in factor_id.lower():
                conditions_mentioned.append(("Meningitis", certainty))
            
            # Add any conditions found
            for condition, cert in conditions_mentioned:
                if condition not in [c["condition"] for c in conditions]:
                    conditions.append({
                        "condition": condition,
                        "certainty": cert
                    })
        
        # Sort by certainty
        conditions.sort(key=lambda x: x["certainty"], reverse=True)
        
        # Add to result
        result["differential_diagnoses"] = conditions
        
        # Add recommended tests based on top diagnoses
        result["recommended_tests"] = generate_recommended_tests(conditions, result.get("entity_id", ""))
    
    return results


def generate_recommended_tests(diagnoses: List[Dict[str, Any]], patient_id: str) -> List[str]:
    """
    Generate recommended tests based on differential diagnoses.
    
    Args:
        diagnoses (list): List of differential diagnoses
        patient_id (str): Patient identifier
        
    Returns:
        list: List of recommended tests
    """
    recommended_tests = []
    
    # Map conditions to recommended tests
    test_map = {
        "Pneumonia": ["Complete Blood Count", "Chest X-ray", "Sputum Culture"],
        "Bacterial Infection": ["Blood Culture", "Procalcitonin", "C-Reactive Protein"],
        "Viral Infection": ["Viral PCR Panel", "Serology Tests"],
        "Acute Coronary Syndrome": ["Cardiac Enzymes", "ECG", "Echocardiogram"],
        "Malaria": ["Malaria Thick and Thin Smears", "Rapid Diagnostic Test", "PCR"],
        "Diabetes Mellitus": ["Fasting Blood Glucose", "HbA1c", "Glucose Tolerance Test"],
        "Meningitis": ["Lumbar Puncture", "Blood Culture", "CT Head"]
    }
    
    # Only include top 3 diagnoses for test recommendations
    for diagnosis in diagnoses[:3]:
        condition = diagnosis["condition"]
        tests = test_map.get(condition, [])
        
        for test in tests:
            if test not in recommended_tests:
                recommended_tests.append(test)
    
    return recommended_tests


def generate_report(results: Dict[str, Any], output_path: Optional[str] = None) -> None:
    """
    Generate a human-readable report from the analysis results.
    
    Args:
        results (dict): Results from hyperlogica processing
        output_path (str, optional): Path where the HTML report should be saved
    """
    # Create HTML report
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Medical Diagnosis Support Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .patient {{ margin-bottom: 30px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .patient h2 {{ color: #444; margin-top: 0; }}
        .patient .details {{ margin-left: 20px; }}
        .PROBABLE_DIAGNOSIS {{ color: red; font-weight: bold; }}
        .UNLIKELY_DIAGNOSIS {{ color: green; font-weight: bold; }}
        .POSSIBLE_DIAGNOSIS {{ color: orange; font-weight: bold; }}
        .demographics {{ margin-top: 10px; color: #666; }}
        .differential {{ margin-top: 15px; }}
        .tests {{ margin-top: 15px; }}
        .certainty {{ font-style: italic; color: #666; }}
        .explanation {{ margin-top: 15px; background-color: #f9f9f9; padding: 10px; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>Medical Diagnosis Support Report</h1>
    <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p><strong>Disclaimer:</strong> This is an AI-assisted analysis for educational purposes only. All medical decisions should be made by qualified healthcare professionals.</p>
    <p>Analyzed {results.get('entities_processed', 0)} patients using {results.get('metadata', {}).get('reasoning_approach', 'unknown')} reasoning approach</p>
    
    <div class="patients">
"""
    
    # Add each patient result
    for patient in results.get("results", []):
        patient_id = patient.get("entity_id", "")
        name = patient.get("entity_name", patient_id)
        demographics = patient.get("demographics", {})
        diagnosis = patient.get("outcome", "UNKNOWN")
        certainty = patient.get("certainty", 0)
        reasoning = patient.get("reasoning", {})
        
        # Format demographics
        demo_text = ""
        if "age" in demographics:
            demo_text += f"{demographics['age']} year old "
        if "sex" in demographics:
            demo_text += f"{demographics['sex']} "
        if "risk_factors" in demographics:
            risk_factors = ", ".join(demographics["risk_factors"])
            demo_text += f"with risk factors: {risk_factors}"
        
        # Format differential diagnoses
        differential = patient.get("differential_diagnoses", [])
        diff_html = "<ul>"
        for diag in differential:
            diff_html += f"<li>{diag['condition']} ({int(diag['certainty'] * 100)}% certainty)</li>"
        diff_html += "</ul>"
        
        # Format recommended tests
        tests = patient.get("recommended_tests", [])
        tests_html = "<ul>"
        for test in tests:
            tests_html += f"<li>{test}</li>"
        tests_html += "</ul>"
        
        explanation = reasoning.get("explanation", "No explanation available.")
        
        html += f"""
    <div class="patient">
        <h2>{name} (ID: {patient_id})</h2>
        <div class="details">
            <p class="demographics">{demo_text}</p>
            
            <p>Diagnostic Assessment: <span class="{diagnosis}">{diagnosis}</span> 
               <span class="certainty">({int(certainty * 100)}% confidence)</span></p>
            
            <div class="differential">
                <h3>Differential Diagnoses:</h3>
                {diff_html}
            </div>
            
            <div class="tests">
                <h3>Recommended Tests:</h3>
                {tests_html}
            </div>
            
            <div class="explanation">
                <h3>Analysis:</h3>
                <p>{explanation}</p>
            </div>
        </div>
    </div>
"""
    
    # Close HTML tags
    html += """
    </div>
</body>
</html>
"""
    
    # Save to file if output path provided
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(html)
        print(f"Report saved to {output_path}")
    else:
        print(html)


def main():
    """Main function to run the medical diagnosis example."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Medical Diagnosis Example using Hyperlogica")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--patients", help="Path to JSON file containing patient records")
    parser.add_argument("--output", help="Path to save output results (JSON)")
    parser.add_argument("--report", help="Path to save HTML report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs("./output", exist_ok=True)
        os.makedirs("./logs", exist_ok=True)
        
        # Load or create configuration
        config = create_or_load_config(args.config)
        
        # Process patients from file if provided
        if args.patients:
            patients = load_patients_from_file(args.patients)
            if patients:
                # Replace the sample entities with loaded patients
                config["input_data"]["entities"] = patients
                print(f"Loaded {len(patients)} patients for analysis")
            else:
                print("No patients loaded, using sample patients")
        
        # Process the configuration with hyperlogica
        print("Starting medical diagnosis analysis...")
        
        hyperlogica_options = {
            "verbose": args.verbose,
            "output_path": args.output if args.output else "./output/medical_diagnosis_results.json",
            "timeout": 600  # 10 minutes
        }
        
        results = process_input_file(
            input_path=None,  # Use in-memory configuration
            options=hyperlogica_options,
            config_dict=config  # Pass configuration directly
        )
        
        # Post-process results to add differential diagnoses and recommended tests
        results = add_differential_diagnoses(results)
        
        # Save updated results
        if args.output or hyperlogica_options["output_path"]:
            output_path = args.output if args.output else hyperlogica_options["output_path"]
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
        
        # Generate report
        report_path = args.report if args.report else "./output/medical_diagnosis_report.html"
        generate_report(results, report_path)
        
        print(f"Analysis complete. Processed {results['entities_processed']} patients.")
        print(f"JSON results saved to {hyperlogica_options['output_path']}")
        print(f"HTML report saved to {report_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
