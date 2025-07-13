from flask import Flask, render_template, request, session, redirect
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.exception import CustomException
import sys
import os
from dotenv import load_dotenv
# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

app = Flask(__name__, static_folder='static', template_folder='templates')

app.secret_key = GEMINI_API_KEY

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
            print(input_df)

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
        return redirect('/')  # fallback

    return render_template('genai.html', result_data=result_data)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
