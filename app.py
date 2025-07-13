from flask import Flask, render_template, request
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.exception import CustomException
import sys

app = Flask(__name__, static_folder='static', template_folder='templates')


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
            prediction = predict_pipeline.predict(input_df)
            predicted_class = prediction[0]

            if predicted_class == 0:
                predicted_class = 'Extrovert'
            elif predicted_class == 1:
                predicted_class = 'Introvert'
            else:
                predicted_class = 'Unknown'

            return render_template('result.html', prediction=predicted_class)
        
        except Exception as e:
            raise CustomException(e, sys)
    
    elif request.method == 'GET':
        return render_template('predict.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
