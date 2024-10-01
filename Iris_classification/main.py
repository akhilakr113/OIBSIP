from flask import *
from prediction import *

app=Flask(__name__)


@app.route('/',methods=['get','post'])
def prediction():
    data={}
    data1=[]
    Car_name=[]
    if 'sub' in request.form:
        sepal_len=request.form['sl']
        sepal_wid=request.form['sw']
        petal_len=request.form['pl']
        petal_wid=request.form['pw']
    
        data=pred(sepal_len,sepal_wid,petal_len,petal_wid)
        


    return render_template('pred.html',data=data)


app.run(debug=True,port=5477)
