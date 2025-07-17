
import google.generativeai as genai

genai.configure(api_key="AIzaSyDjY9WaQinE1rbn1yI6U9dp2JWdPL_jYxk")

gemini_model = genai.GenerativeModel("models/gemini-1.5-flash")

def burn_ai_assistant(user_input):
    prompt = (
        "You are a certified dermatologist assistant. Provide helpful, safe, and clear responses "
        "related to burn injuries, skin treatment, wound care, and model explanations.\n"
        f"Patient says: {user_input}"
    )
    response = gemini_model.generate_content(prompt)
    return response.text.strip()
