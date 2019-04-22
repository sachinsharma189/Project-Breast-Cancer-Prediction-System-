from flask import Flask, render_template, request
from sklearn.metrics import confusion_matrix, classification_report, precision_score
import patsy

app = Flask(__name__)

import _pickle as pickle
import pandas as pd

@app.route('/')
def student():
   return render_template('student.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
      result = request.form
      inputDict = {}
      inputDict['diagnosis'] = 'B'
      inputDict.update(result.to_dict(flat=True))
      # import pdb;pdb.set_trace()
      df = pd.DataFrame(inputDict, index=[0])
      print("Dataframe:",df)
      print("Shape of df" , df.shape)

      fob = open("model_fit", "rb")
      logistic_fit = pickle.load(fob)
      print('Logistic fit fileeeeeeee',logistic_fit)
      fob2 = open("Xtest", "rb")
      X_test = pickle.load(fob2)
      print(X_test,'111111111111111111111')
      print("Shape of X_test" , X_test.shape)
      x,y= patsy.dmatrices("diagnosis ~ radius_mean + texture_mean + smoothness_mean + compactness_mean + symmetry_mean + fractal_dimension_mean + radius_se + texture_se + smoothness_se + compactness_se + symmetry_se + fractal_dimension_se", data=df, return_type='dataframe')  # df is data for prediction
      # newdf=pd.DataFrame({'category':['a','b']*3})
      # print(newdf,'nnnnewwwwwwwwwwwwddddddddddfffffffff')
      # x= patsy.dmatrix('0 + category', data=newdf)
      print(x,'xxxxxxx')
      print("Shape of x" , x.shape)
      print(y,'yyyyyyyyy')
      print("Shape of y" , y.shape)
      predictions = logistic_fit.predict(x, transform=True)
      # predictions[1:6]
      # print(predictions)
      # predictions_nominal = [ "M" if x < 0.5 else "B" for x in predictions]
      # predictions_nominal[1:6]
      # print(classification_report(y_test, predictions_nominal, digits=3))

      # cfm = confusion_matrix(y_test, predictions_nominal)

      # true_negative = cfm[0][0]
      # false_positive = cfm[0][1]
      # false_negative = cfm[1][0]
      # true_positive = cfm[1][1]

      # print('Confusion Matrix: \n', cfm, '\n')

      # print('True Negative:', true_negative)
      # print('False Positive:', false_positive)
      # print('False Negative:', false_negative)
      # print('True Positive:', true_positive)
      # print('Correct Predictions', 
      # round((true_negative + true_positive) / len(predictions_nominal) * 100, 1), '%')

      return render_template("result.html" ,result = result,)

if __name__ == '__main__':
   app.run(debug = True)