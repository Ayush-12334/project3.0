import sys
import pandas as pd 
from src.exception import CustomException
from src.logger import logging
from src.components .data_transformation import DataTransformation
from src.utils import load_object

class PredictionPipeline:
    def __init__(self):
        pass
    def predict(self,feature):
        try:
            model_path="artifacts/model.pkl"
            preprocessor_path="artifacts/preprocessor.pkl"
            model=load_object(file_path=model_path)
            
          
            tranformation=DataTransformation()
            feature=tranformation.feature_engineering(feature)
            preprocessor=load_object(file_path=preprocessor_path)
            
            scaled_data=preprocessor.transform(feature)
            preds=model.predict(scaled_data)
            return preds
        except Exception as e:
            raise CustomException(e,sys)
        






class CustomData:
    def __init__(
        self,
        year_birth,
        education,
        marital_status,
        income,
        kidhome,
        teenhome,
        Dt_Customer,
        recency,
        mnt_wines,
        mnt_fruits,
        mnt_meat,
        mnt_fish,
        mnt_sweet,
        mnt_gold,
        num_deals,
        num_web,
        num_catalog,
        num_store,
        num_web_visits,
        accepted_cmp3,
        accepted_cmp4,
        accepted_cmp5,
        accepted_cmp1,
        accepted_cmp2,
        complain
    ):
        try:
            self.year_birth = year_birth
            self.education = education
            self.marital_status = marital_status
            self.income = income
            self.kidhome = kidhome
            self.teenhome = teenhome
            self.Dt_Customer= Dt_Customer
            self.recency = recency
            self.mnt_wines = mnt_wines
            self.mnt_fruits = mnt_fruits
            self.mnt_meat = mnt_meat
            self.mnt_fish = mnt_fish
            self.mnt_sweet = mnt_sweet
            self.mnt_gold = mnt_gold
            self.num_deals = num_deals
            self.num_web = num_web
            self.num_catalog = num_catalog
            self.num_store = num_store
            self.num_web_visits = num_web_visits
            self.accepted_cmp3 = accepted_cmp3
            self.accepted_cmp4 = accepted_cmp4
            self.accepted_cmp5 = accepted_cmp5
            self.accepted_cmp1 = accepted_cmp1
            self.accepted_cmp2 = accepted_cmp2
            self.complain = complain

        except Exception as e:
            raise Exception(f"Error in CustomData initialization: {str(e)}")

    def get_data_as_dataframe(self):
        """
        Converts user input into DataFrame
        """
        
        try:
            return pd.DataFrame({
                "Year_Birth": [self.year_birth],
                "Education": [self.education],
                "Marital_Status": [self.marital_status],
                "Income": [self.income],
                "Kidhome": [self.kidhome],
                "Teenhome": [self.teenhome],
                "Dt_Customer":[self.Dt_Customer],
                "Recency": [self.recency],
                "MntWines": [self.mnt_wines],
                "MntFruits": [self.mnt_fruits],
                "MntMeatProducts": [self.mnt_meat],
                "MntFishProducts": [self.mnt_fish],
                "MntSweetProducts": [self.mnt_sweet],
                "MntGoldProds": [self.mnt_gold],
                "NumDealsPurchases": [self.num_deals],
                "NumWebPurchases": [self.num_web],
                "NumCatalogPurchases": [self.num_catalog],
                "NumStorePurchases": [self.num_store],
                "NumWebVisitsMonth": [self.num_web_visits],
                "AcceptedCmp3": [self.accepted_cmp3],
                "AcceptedCmp4": [self.accepted_cmp4],
                "AcceptedCmp5": [self.accepted_cmp5],
                "AcceptedCmp1": [self.accepted_cmp1],
                "AcceptedCmp2": [self.accepted_cmp2],
                "Complain": [self.complain],
                # Add missing columns with default values
                "Response": [0]  # default 0 if
            })

        except Exception as e:
            raise Exception(f"Error converting input to DataFrame: {str(e)}")
