import ast
import pickle
apps_lines = tuple(open("data/amazon/apps.json", 'r'))
app_json = []
id1 = 0
for line in apps_lines:
    tempdict = ast.literal_eval(line)
    sentiment = 0
    if tempdict["overall"] >= 3:
        sentiment = 1
    else:
        sentiment = 0
    app_json.append({'id' : id1,'sentiment':sentiment, 'review': tempdict['reviewText']})
    id1 += 1
print(app_json[0])
pickle.dump(app_json, open("app.pickle", "wb"))
mobile_lines = tuple(open("data/amazon/mobile.json", 'r'))
mobile_json = []
for line in mobile_lines:
    tempdict = ast.literal_eval(line)
    sentiment = 0
    if tempdict["overall"] >= 3:
        sentiment = 1
    else:
        sentiment = 0
    mobile_json.append({'id' : id1,'sentiment':sentiment, 'review': tempdict['reviewText']})
    id1 += 1
print(mobile_json[0])
pickle.dump(mobile_json, open("mobile.pickle", "wb"))
