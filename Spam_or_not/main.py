from flask import *
from prediction import *

app=Flask(__name__)


@app.route('/',methods=['get','post'])
def prediction():
    data={}
    data1=[]
    Car_name=[]
    if 'sub' in request.form:
        text=request.form['text']
     

        data=spam_or_not(text)
        


    return render_template('pred.html',data=data,Car_name=Car_name)


app.run(debug=True,port=5468)
