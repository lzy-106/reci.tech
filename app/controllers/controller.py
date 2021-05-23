# Controller for the backend logic
from flask import *
from app import app
from .audio_inference import infer_from_audio

emotion_dict = {
    0: 'angry',
    1: 'calm',
    2: 'fearful',
    3: 'happy',
    4: 'sad',
}


@app.route("/")
def index():
    """
    The homepage for our beautiful stunning website.
    """
    return render_template("app/home.html")


@app.route("/speak")
def speak():
    """
    The page for the user to input their voice.
    """
    data = {'script': request.args.get('script')}
    session['script'] = data['script']
    return render_template("app/speak.html", data=data)


@app.route("/uploadLine", methods=["POST"])
def uploadLine():
    """
    The upload page for our beautiful stunning website with a upload button
    for the chosen play.
    """
    if request.method == "POST":
        script = request.form['script']
        return redirect(url_for('speak', script=script))


@app.route("/uploadAudio", methods=["POST"])
def uploadAudio():
    if request.method == "POST":
        f = request.files['audio_data']
        with open('audio.wav', 'wb') as audio:
            f.save(audio)
            try:
                suggested_sentiment, user_sentiment \
                    = runModel('audio.wav', session['script'])
            except ValueError as error:
                print('Error from ML model:', error)
                return redirect(url_for('index')), 400

            session['suggested_sentiment'] = suggested_sentiment
            session['user_sentiment'] = str(user_sentiment)
            if suggested_sentiment in user_sentiment.keys():
                session['accuracy'] = str(user_sentiment[suggested_sentiment])
            else:
                session['accuracy'] = 'sadly, 0'
            return redirect(url_for('showResult'))


@app.route("/showResult")
def showResult():
    data = {}
    data['suggested_sentiment'] = session['suggested_sentiment']
    data['user_sentiment'] = session['user_sentiment']
    data['accuracy'] = session['accuracy']
    return render_template("app/result.html", data=data)


# @app.route("/runModel", methods=["POST"])
def runModel(audio, script):
    # TODO text sentiment inference
    suggested_sentiment_index = 0
    user_sentiment_index = infer_from_audio(audio)
    # user_sentiment_index = [(0, 40.5), (4, 14.9), (3, 14.9)]
    user_sentiment = {emotion_dict[i[0]]: i[1] for i in user_sentiment_index}
    return emotion_dict[suggested_sentiment_index], user_sentiment
