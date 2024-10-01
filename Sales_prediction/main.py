from flask import *
from ss import *

app=Flask(__name__)


@app.route('/',methods=['get','post'])
def prediction():
    data={}
    data1=[]
    Car_name=[]
    if 'sub' in request.form:
        tv=request.form['tv']
        radio=request.form['radio']
        np=request.form['np']


     

        data=prediction_sale(tv,radio,np)
        


    return render_template('pred.html',data=data,Car_name=Car_name)


app.run(debug=True,port=5490)
