from sklearn.externals import joblib 
from flask import Flask,render_template,url_for,flash,redirect,request,jsonify
app=Flask(__name__)
app.config['SECRET_KEY']='24fbcde8e1de3d3404d190895b9b4093'
@app.route('/')
def home():
    return render_template("view.html")
@app.route("/",methods=['POST'])
def jsondata():
    a=float(request.form['slength'])
    b=float(request.form['swidth'])
    c=float(request.form['plength'])
    d=float(request.form['pwidth'])
  
    arr=[[a,b,c,d]]
    p=importModel(arr)
    return str(p);
def importModel(X_test):

    model_from_joblib = joblib.load('filename.pkl')  
    p=model_from_joblib.predict(X_test)
    p=int(p)
    dict={0:'Iris-setosa',1:'Iris-versicolor',2:'Iris-virginica'}
    return dict[p]

   
if __name__ == '__main__':
    app.run(debug=True, port=5000)