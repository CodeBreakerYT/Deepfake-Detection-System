import os
import google.generativeai as genai
from PIL import Image
import json

class VLMAnalyzer:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.enabled = bool(self.api_key)
        if self.enabled:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.5-flash', generation_config={"response_mime_type": "application/json"})
        else:
            print("VLMAnalyzer disabled: GEMINI_API_KEY not found in .env")

    def analyze_frame(self, image_array):
        if not self.enabled:
            return None
        
        try:
            pil_img = Image.fromarray(image_array).convert("RGB")
            w, h = pil_img.width, pil_img.height
            
            prompt = f"""
You are an expert deepfake detection AI.
Analyze the provided image for AI-generation artifacts, unnatural physics, logical inconsistencies, face swaps, or morphed hands/limbs.

Return a JSON strictly following this schema:
{{
  "is_ai_generated": boolean,
  "semantic_fake_score": float (0.0 to 1.0),
  "reasoning": string (Detailed explanation of WHY it is fake based on physics, anatomy, or context. e.g. "The dog is walking on two feet". If real, explain why.),
  "anomaly_regions": [
    {{
      "box": [x, y, w, h] (approximate integer pixel coordinates relative to image width {w} and height {h}),
      "label": string (e.g., "morphed hand", "unnatural posture")
    }}
  ]
}}
"""
            response = self.model.generate_content([prompt, pil_img])
            
            text = response.text
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].strip()
                
            return json.loads(text)
        except Exception as e:
            print(f"VLM Analysis Error: {e}")
            return None
