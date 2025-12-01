# handler.py

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
# Import accelerate and bitsandbytes are implicitly used by transformers
# once they are installed via requirements.txt

# --- Configuration ---
# Use environment variable for model ID for easy updates, otherwise use a default
MODEL_ID = os.environ.get("MODEL_ID", "SURENKUMAAR/1.3b_modelsuren") 
# ^^^ REPLACE "your/model/path" with your actual Hugging Face Model ID ^^^
# ---------------------

class ModelHandler:
    """
    Handles model loading and inference for the RunPod worker.
    The model is loaded in __init__ (cold start).
    """
    def __init__(self):
        # 1. Load the model immediately when the container starts
        self.model, self.tokenizer = self._load_model()
        print("Model Initialization Complete.")

    def _load_model(self):
        print(f"Loading Model: {MODEL_ID}...")
        
        # --- QUANTIZATION LOGIC STARTS HERE ---
        # This reduces the model size and improves cold start time
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            # Instruction for 4-bit Quantization (The key for speed)
            load_in_4bit=True,
            # Data type for high-quality computation
            torch_dtype=torch.bfloat16,
            # Automatically manage model placement on GPU(s)
            device_map="auto"
        )
        # --- QUANTIZATION LOGIC ENDS HERE ---
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        print("Model Loaded Successfully with 4-bit Quantization!")
        return model, tokenizer

    def run_inference(self, job_input):
        """
        This method is called for every API request to your RunPod endpoint.
        It takes the input from the Q&A website and returns the model's answer.
        """
        # Get the prompt (question) from the job input
        prompt = job_input.get("prompt", "What is the capital of Malaysia?")
        
        print(f"Received Prompt: {prompt}")

        # --- YOUR Q&A INFERENCE LOGIC GOES HERE ---
        # 1. Tokenize the input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # 2. Generate the output
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=256, 
                do_sample=True,
                temperature=0.7 
                # Add/adjust generation parameters as needed
            )
            
        # 3. Decode the output
        response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # ------------------------------------------

        # Return the result as expected by the RunPod worker
        return {"result": response_text}

# Placeholder for RunPod worker startup (often handled by the environment)
if __name__ == "__main__":
    # This part is typically handled by the RunPod worker setup.
    # It’s included here only for structural completeness.
    print("RunPod Worker starting...")
    # The ModelHandler will be initialized by the worker process.
    # To test locally: handler = ModelHandler()
