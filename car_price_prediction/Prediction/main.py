from flask import *
from ss import *

app=Flask(__name__)


@app.route('/',methods=['get','post'])
def prediction():
    data={}
    data1=[]
    Car_name=[]
    if 'sub' in request.form:
        year=request.form['year']
        Prsantprice=request.form['pp']
        Kilometer_driven=request.form['kd']
        Fuel_type=request.form['ft']
        Seller_type=request.form['st']
        Owner=request.form['owner']
        Car_name=request.form['cname']

        data=prediction_price(year,Prsantprice,Kilometer_driven,Fuel_type,Seller_type,Owner,Car_name)
        


    return render_template('pred.html',data=data,Car_name=Car_name)


app.run(debug=True,port=5466)
