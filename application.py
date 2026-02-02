from flask import Flask ,request,render_template
import pandas as pd 
import numpy as np
from src.pipelines.prediction_pipeline import PredictionPipeline,CustomData
from sklearn.preprocessing import StandardScaler


app=Flask(__name__)



## Route For the home page 



@app.route("/")

def index():
    return render_template("index.html")


@app.route('/prediction',methods=['GET','POST'])
def prediction_datapoint():
    if request.method=='GET':
          return render_template('home.html')
    else:
        # 👇 request.form IS a dict-like object
        data = CustomData(
            year_birth = request.form.get('year_birth'),
            education = request.form.get('education'),
            marital_status = request.form.get('marital_status'),
            income = request.form.get('income'),
            kidhome = request.form.get('kidhome'),
            teenhome = request.form.get('teenhome'),
            Dt_Customer=request.form.get('Dt_Customer'),
            recency = request.form.get('recency'),
            mnt_wines = request.form.get('mnt_wines'),
            mnt_fruits = request.form.get('mnt_fruits'),
            mnt_meat = request.form.get('mnt_meat'),
            mnt_fish = request.form.get('mnt_fish'),
            mnt_sweet = request.form.get('mnt_sweet'),
            mnt_gold = request.form.get('mnt_gold'),
            num_deals = request.form.get('num_deals'),
            num_web = request.form.get('num_web'),
            num_catalog = request.form.get('num_catalog'),
            num_store = request.form.get('num_store'),
            num_web_visits = request.form.get('num_web_visits'),
            accepted_cmp3 = request.form.get('accepted_cmp3'),
            accepted_cmp4 = request.form.get('accepted_cmp4'),
            accepted_cmp5 = request.form.get('accepted_cmp5'),
            accepted_cmp1 = request.form.get('accepted_cmp1'),
            accepted_cmp2 = request.form.get('accepted_cmp2'),
            complain = request.form.get('complain')
        )
        pred_df=data.get_data_as_dataframe()
        print(pred_df)

        prediction_pipeline=PredictionPipeline()
        results=prediction_pipeline.predict(pred_df)
        print(pred_df)
        prediction=int(results[0])

        cluster = {
                0: "This customer is not a high-value asset",
                1: "This customer buys a decent amount of products",
                2: "This is a high-value customer"
            }


        final_prediction=cluster.get(prediction,'unknown customer')

        return render_template("home.html",results=final_prediction)


if __name__=="__main__":
     app.run()