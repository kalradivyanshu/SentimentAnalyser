from sentimentanalyzer import predict
from flask import Flask, request, render_template

app = Flask(__name__)

@app.route("/")
def index():
	return render_template('index.html')

@app.route("/predict/", methods=['GET', 'POST'])
def makeprediction():
	if request.method == 'GET':
		tweet = request.args.get('tweet', None)
		prediction = predict(tweet)
		return str((prediction[0][0][0], prediction[0][0][1]))
		#return str((0.0, 0.0))

if __name__ == "__main__":
	app.run(debug=True, host='0.0.0.0')