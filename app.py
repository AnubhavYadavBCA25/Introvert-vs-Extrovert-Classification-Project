from flask import Flask, render_template, request, session, redirect, url_for, jsonify
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.exception import CustomException
import sys
import os
from dotenv import load_dotenv
import google.generativeai as genai
import markdown
# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=GEMINI_API_KEY)

# Safety Settings
safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
]

# Generation Configurations
generation_config = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 30,
    "max_output_tokens": 1000,
    "response_mime_type": "text/plain"
}

system_instruction = (
        "You are an expert personality therapist. Based on the user's personality traits and prediction, "
        "suggest 5 personalized activities, habits, or tips that can help the user either embrace or develop their personality. "
        "Provide short, practical, and empathetic points. Make sure the tone is positive and growth-focused.\n"
        "Format it in simple bullet points.\n\n" \
        "--> Format the output in proper Markdown with:\n"
        "- Bullet points using '*'\n"
        "- Use **bold** for important words\n"
        "- No numbering\n"
    )

# Initialize Flask app and setup API key
app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = GEMINI_API_KEY

# Gemini GenAI Model Configuration
model = genai.GenerativeModel(
    model_name='gemini-1.5-flash',
    generation_config=generation_config,
    safety_settings=safety_settings,
    system_instruction=system_instruction
)

model_chat_bot = genai.GenerativeModel(
    model_name='gemini-1.5-flash',
    generation_config=generation_config,
    safety_settings=safety_settings
)

# Home Route - Index Page
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        try:
            data = CustomData(
                time_spent_alone=float(request.form.get("time_spent_alone")),
                stage_fear=str(request.form.get("stage_fear")),
                social_event_attendance=int(request.form.get("social_event_attendance")),
                going_outside=int(request.form.get("going_outside")),
                drained_after_socializing=str(request.form.get("drained_after_socializing")),
                friends_circle_size=int(request.form.get("friends_circle_size")),
                post_frequency=int(request.form.get("post_frequency"))
            )

            input_df = data.get_data_as_df()
            # print(input_df)

            predict_pipeline = PredictPipeline()
            prediction = predict_pipeline.predict(input_df)[0]
            
            result_data = input_df.copy()
            result_data["Prediction"] = prediction

            # 5. Store for next page (use session or global var)
            session["result_data"] = result_data.to_dict(orient="records")[0]
            print(result_data)

            return render_template('result.html', result=result_data.to_dict(orient="records")[0])
        
        except Exception as e:
            raise CustomException(e, sys)
    
    return render_template('predict.html')

@app.route('/genai', methods=['GET', 'POST'])
def genai():
    result_data = session.get("result_data", None)
    if result_data is None:
        return redirect(url_for('home', showpopup=1))  # fallback

    suggestions = ""

    try:
        #Prompt generation
        user_input_summary = "\n".join([
            f"{key.replace('_', ' ')}: {value}"
            for key, value in result_data.items() if key != "Prediction"
        ])
        personality = result_data["Prediction"]

        prompt = f"User is predicted to be an '{personality}'.\nHere are the details:\n{user_input_summary}. \nDon't mention the prediction 1 or 0, just mention the personality type. \nSample response should the suggestions don't mention based on inputs.\n\n{system_instruction}"
        # print(f"Prompt for Gemini: {prompt}")

        # ✅ Use list format — Gemini prefers this
        response = model.generate_content([prompt])
        
        if response and hasattr(response, 'text'):
            suggestions = markdown.markdown(response.text.strip())
            print("Response from Gemini:\n", suggestions)
        else:
            print("⚠️ Gemini returned no usable text.")
            suggestions = "⚠️ AI did not return any suggestions. Please try again."

    except Exception as e:
        print(f"Error generating suggestions: {e}")
        suggestions = "⚠️ Unable to generate suggestions at the moment. Please try again later."

    # --- Final Step: Return template with suggestions ---
    return render_template('genai.html', result_data=result_data, suggestions=suggestions)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        # Get user message
        user_input = request.json.get('message')
        if not user_input:
            return jsonify({'response': "No input received."}), 400

        # Get user context
        user_data = session.get("result_data", None)
        if not user_data:
            return jsonify({'response': "Prediction data missing. Please complete the prediction first."}), 400

        # Build context string
        personality = "Introvert 🧍‍♂️" if user_data["Prediction"] == 1 else "Extrovert 🗣️"
        context_info = f"""
        The user is a {personality}.
        Time Spent Alone: {user_data["Time_spent_Alone"]} hrs/day,
        Stage Fear: {user_data["Stage_fear"]},
        Social Events/Week: {user_data["Social_event_attendance"]},
        Going Outside: {user_data["Going_outside"]},
        Drained After Socializing: {user_data["Drained_after_socializing"]},
        Friends Circle Size: {user_data["Friends_circle_size"]},
        Post Frequency: {user_data["Post_frequency"]}
        """

        # System instruction
        prompt = f"""
            You are a friendly AI mental well-being assistant trained to support personality-based interaction.

            Based on the user's personality and behavior traits:

            {context_info}

            User says: "{user_input}"

            Now respond in a kind, helpful, and psychologically-aware tone. Keep the response concise and conversational.

            Don't provide suggestions or advice unless explicitly asked. Focus on understanding and engaging with the user's input.
            """

        response = model_chat_bot.generate_content(prompt)
        bot_reply = markdown.markdown(response.text.strip())

        return jsonify({'response': bot_reply})

    except Exception as e:
        return jsonify({'response': f"Something went wrong: {str(e)}"}), 500

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
