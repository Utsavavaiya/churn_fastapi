import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import openai
import os
from dotenv import load_dotenv

load_dotenv()

class ChurnExplanationGenerator:
    def __init__(self, provider='openai', api_key=None):
        """
        Initialize the explanation generator.
        
        Args:
            provider: "openai", "huggingface", or "local"
            api_key: API key for the chosen provider
        """

        self.provider = provider
        if provider == "openai":
            if api_key:
                openai.api_key = api_key
            else:
                openai.api_key = os.getenv("OPENAI_API_KEY")

    def create_prompt(self, row_data, prediction, probability=None, feature_names=None):
        """
        Create a prompt for GenAI explanation.
        
        Args:
            row_data: Dictionary of customer features
            prediction: Predicted class (0/1 or "Yes"/"No")
            probability: Prediction probability (optional)
            feature_names: List of feature names
        
        Returns:
            String prompt for GenAI
        """

        # Converting prediction to human-readable format
        churn_status = "churn" if prediction == 1 or prediction == "Yes" else "not churn"


        # Extract key features for explanation
        key_features = self._extract_key_features(row_data)

        # Build probability text
        prob_text = ""
        if probability is not None:
            confidence = max(probability)*100
            prob_text = f" with {confidence:.1f}% confidence"

        prompt = f"""
        Given the following customer data, explain why this customer is predicted to {churn_status}{prob_text}.

        Customer Features:
        {key_features}

        Prediction: {churn_status.title()}

        Provide a concise, business-friendly explanation in 1-2 sentences focusing on the most important factors that led to this prediction. Use language that a business manager would understand.
        """

        return prompt

    def _extract_key_features(self, row_data):
        """
        Extract and format key features for the prompt.
        """
        feature_text = ""

        # Handle common churn-related features
        key_feature_mapping = {
            'tenure': 'Tenure (months)',
            'monthlycharges': 'Monthly Charges',
            'totalcharges': 'Total Charges',
            'age': 'Age',
            'usage_frequency': 'Usage Frequency',
            'customer_service_calls': 'Customer Service Calls',
            'contract': 'Contract Type',
            'gender': 'Gender',
            'senior_citizen': 'Senior Citizen'
        }

        for feature, display_name in key_feature_mapping.items():
            # Look for the feature in various formats
            for col in row_data.keys():
                if feature.lower() in col.lower():
                    value = row_data[col]
                    if isinstance(value, (int, float)):
                        feature_text += f"- {display_name}: {value}\n"
                    else:
                        feature_text += f"- {display_name}: {value}\n"
                    break

        # Add binary features (one-hot encoded)
        binary_features = []
        for col, value in row_data.items():
            if value == 1 and ('_' in col or col.startswith(('is', 'has'))):
                binary_features.append(col.replace('_', ' ').title())
        
        if binary_features:
            feature_text += f"- Active Features: {', '.join(binary_features[:5])}\n"
        
        return feature_text.strip()

    def generate_explanation(self, row_data, prediction, probability=None, feature_names=None):
        """
        Generate natural language explanation for a prediction.
        
        Args:
            row_data: Dictionary of customer features
            prediction: Predicted class
            probability: Prediction probability
            feature_names: List of feature names
        
        Returns:
            String explanation
        """
        if self.provider == "openai":
            return self._generate_openai_explanation(row_data, prediction, probability, feature_names)
        elif self.provider == "huggingface":
            return self._generate_huggingface_explanation(row_data, prediction, probability, feature_names)
        else:
            return self._generate_simple_explanation(row_data, prediction, probability)

        
    import openai

    def _generate_openai_explanation(self, row_data, prediction, probability, feature_names):
        """Generate explanation using OpenAI API."""
        try:
            prompt = self.create_prompt(row_data, prediction, probability, feature_names)

            response = openai.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": "You are a data analyst explaining customer churn predictions to business stakeholders."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.5
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"OpenAI API error: {e}")
            return self._generate_simple_explanation(row_data, prediction, probability)

    
    def _generate_huggingface_explanation(self, row_data, prediction, probability, feature_names):
        """Generate explanation using HuggingFace (placeholder for now)."""
        
        return self._generate_simple_explanation(row_data, prediction, probability)
    
    def _generate_simple_explanation(self, row_data, prediction, probability):
        """Generate a simple rule-based explanation as fallback."""
        churn_status = "likely to churn" if prediction == 1 or prediction == "Yes" else "unlikely to churn"
        
        # Simple heuristic-based explanation
        factors = []
        
        # Check for common churn indicators
        for col, value in row_data.items():
            if 'tenure' in col.lower() and isinstance(value, (int, float)):
                if value < 12:
                    factors.append("short tenure")
            elif 'charges' in col.lower() and isinstance(value, (int, float)):
                if value > 80:
                    factors.append("high monthly charges")
            elif 'contract' in col.lower() and 'month' in str(value).lower():
                factors.append("month-to-month contract")
        
        if factors:
            explanation = f"Customer is {churn_status} due to {', '.join(factors[:3])}."
        else:
            explanation = f"Customer is {churn_status} based on their profile characteristics."
        
        if probability is not None:
            confidence = max(probability) * 100
            explanation += f" (Confidence: {confidence:.1f}%)"
        
        return explanation    


        