import os
import requests
import openai
import json
from dotenv import load_dotenv

# Force load config
load_dotenv("env_config.txt")

class LLMProvider:
    def __init__(self, provider="huggingface"):
        self.provider = provider
        self.api_key = os.getenv("HUGGINGFACE_API_KEY") if provider == "huggingface" else os.getenv("OPENAI_API_KEY")
        self.model_name = os.getenv("HUGGINGFACE_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
        
        # DEBUG: Verify initialization
        key_status = "SET" if self.api_key else "MISSING"
        print(f"DEBUG: LLMProvider Initialized. Provider={provider}, Model={self.model_name}, Key={key_status}")
        if self.api_key:
            print(f"DEBUG: Key Preview: {self.api_key[:5]}...")

    def generate(self, prompt, system_prompt="You are a helpful assistant."):
        if self.provider == "openai":
            return self._generate_openai(prompt, system_prompt)
        elif self.provider == "huggingface":
            return self._generate_huggingface(prompt, system_prompt)
        else:
            return "Error: Unknown provider."

    def _generate_openai(self, prompt, system_prompt):
        try:
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"OpenAI Error: {str(e)}"

    def _generate_huggingface(self, prompt, system_prompt):
        """
        Uses HuggingFace Serverless Inference API via official client.
        """
        if not self.api_key:
            return "Error: HUGGINGFACE_API_KEY not found in environment."

        try:
            from huggingface_hub import InferenceClient
            client = InferenceClient(api_key=self.api_key)
            
            messages = [
                {"role": "user", "content": f"{system_prompt}\n\n{prompt}"}
            ]

            response = client.chat.completions.create(
                model=self.model_name, 
                messages=messages, 
                max_tokens=1024,
                temperature=0.7
            )
            return response.choices[0].message.content
            
        except Exception as e:
            return f"HuggingFace Client Error: {str(e)}"


