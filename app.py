# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 12:45:55 2020

@author: Madhur
"""

from flask import Flask, request
import numpy as np
import pickle
import pandas as pd



app=Flask(__name__)

pickle_in = open("pickle_model.pkl","rb")
classifier=pickle.load(pickle_in)

@app.route('/')
def welcome():
    return "Welcome All"

@app.route('/predict',methods=["Get"])
def predict_note_authentication():
    
    ApprovedLoan=request.args.get("ApprovedLoan")
    Age=request.args.get("Age")
    Bank=request.args.get("Bank")
    Birth_City_Freq=request.args.get("Birth_City_Freq")
    Gender_Male_Female_encode=request.args.get("Gender_Male_Female_encode")
    Has_Registered_phone_Number=request.args.get("Has_Registered_phone_Number")
    Has_family_members_Registered=request.args.get("Has_family_members_Registered")
    How_many_in_punishment=request.args.get("How_many_in_punishment")
    Legal_Accounts=request.args.get("Legal_Accounts")
    Overdue_Accounts=request.args.get("Overdue_Accounts")
    account_in_back=request.args.get("account_in_back")
    approximate_income=request.args.get("approximate_income")
    as_us_contact_Freq=request.args.get("as_us_contact_Freq")
    client_type=request.args.get("client_type")
    closed_account=request.args.get("closed_account")
    company_months=request.args.get("company_months")
    normal_account=request.args.get("normal_account")
    principal_amount=request.args.get("principal_amount")
    purpose_credit_Freq=request.args.get("purpose_credit_Freq")
    salary=request.args.get("salary")
    term_frequency=request.args.get("term_frequency")
         
    prediction_arr=classifier.predict_proba([[float(ApprovedLoan),float(Age),float(Bank),float(Birth_City_Freq),float(Gender_Male_Female_encode),float(Has_Registered_phone_Number),float(Has_family_members_Registered),float(How_many_in_punishment),float(Legal_Accounts),float(Overdue_Accounts),float(account_in_back),float(approximate_income),float(as_us_contact_Freq),float(client_type),float(closed_account),float(company_months),float(normal_account),float(principal_amount),float(purpose_credit_Freq),float(salary),float(term_frequency)]])

    prediction=np.array_split(prediction_arr,2)
    output_0 = prediction[0][0][0]*100
    output_1 = prediction[0][0][1]*100

    return 'Percentage of Customer being Zero i.e Default Loan is {}% \n Percentage of Customer being one i.e OK or GOOD is {}%'.format(output_0,output_1)

@app.route('/predict_file',methods=["POST"])
def predict_note_file():
    
    df_test=pd.read_csv(request.files.get("file"))
    print(df_test.head())
    
    prediction_arr = classifier.predict_proba(df_test)
    prediction=np.array_split(prediction_arr,2)
    output_0 = prediction[0][0][0]*100
    output_1 = prediction[0][0][1]*100

    return 'Percentage of Customer being Zero i.e Default Loan is {}% \n Percentage of Customer being one i.e OK or GOOD is {}%'.format(output_0,output_1)



if __name__=='__main__':
    app.run()