

"""
To run this: streamlit run App.py
Please run attached file Flow.py
"""

# import libraries
import streamlit as st
from metaflow import Flow
from metaflow import get_metadata, metadata
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from App_utils import cleanStatement,get_sentiment_from_pretrained_model, get_sentiment_score_LM
from Flow_utils import feature_preprocess


# make sure we point the app to the flow folder
# NOTE: MAKE SURE TO RUN THE FLOW AT LEAST ONE BEFORE THIS ;-)
FLOW_NAME = 'Multiclass_RandomForest' # name of the target class
# Set the metadata provider as the src folder in the project,
# which should contains /.metaflow
metadata('../src')
# Fetch currently configured metadata provider - check it's local!
print(get_metadata())

# build up the dashboard
st.markdown("# Predicting Federal Funds Rate using a Multiclass_RandomForest")
st.write("Our model uses economic data from St. Louis Fed, dating back to 1994")
st.write("Our model uses actual economic data points and also percent change from prior/last data")

@st.cache_resource
@st.cache_data
def get_latest_successful_run(flow_name: str):
    "Gets the latest successfull run."
    for r in Flow(flow_name).runs():
        if r.successful: 
            return r

# get artifacts from latest run, using Metaflow Client API
latest_run = get_latest_successful_run(FLOW_NAME)
latest_Xtest = latest_run.data.X_test
full_df = latest_run.data.X
y_predicted = latest_run.data.y_pred
y_test = latest_run.data.y_test

latest_model = latest_run.data.last_rf_model

st.markdown("## Sample Dataset")
# show dataset
st.write(full_df.head(2))
#st.write(last_record)

target = ['Target_category']
num_features = ['PCE','Consumer_Sentiment','CPI','ISM PMI','Housing_starts','Unemp_rate','SPX','10y_try','sentiment_score_LM']
cat_features = ['Sentiment']
raw_features_selected = ['Target_category','PCE','Consumer_Sentiment','CPI','ISM PMI','Housing_starts','Unemp_rate','SPX','10y_try','sentiment_score_LM','Sentiment']
selected_features = latest_run.data.selected_features

historical_records = latest_run.data.df_all2[raw_features_selected]

def main():
    # 初始化 Session State
    if 'sentiment_done' not in st.session_state:
        st.session_state.sentiment_done = False
    if 'sentiment' not in st.session_state:
        st.session_state.sentiment = None
    if 'sentiment_score_lm' not in st.session_state:
        st.session_state.sentiment_score_lm = None

    st.header('Input FOMC Statement for Sentiment Analysis')
    statement_sample = '''
            For release at 2:00 p.m. EDT
            Share
            Recent indicators suggest that economic activity expanded at a strong pace in the third quarter. Job gains have moderated since earlier in the year but remain strong, and the unemployment rate has remained low. Inflation remains elevated.
            The U.S. banking system is sound and resilient. Tighter financial and credit conditions for households and businesses are likely to weigh on economic activity, hiring, and inflation. The extent of these effects remains uncertain. The Committee remains highly attentive to inflation risks.
            The Committee seeks to achieve maximum employment and inflation at the rate of 2 percent over the longer run. In support of these goals, the Committee decided to maintain the target range for the federal funds rate at 5-1/4 to 5-1/2 percent. The Committee will continue to assess additional information and its implications for monetary policy. In determining the extent of additional policy firming that may be appropriate to return inflation to 2 percent over time, the Committee will take into account the cumulative tightening of monetary policy, the lags with which monetary policy affects economic activity and inflation, and economic and financial developments. In addition, the Committee will continue reducing its holdings of Treasury securities and agency debt and agency mortgage-backed securities, as described in its previously announced plans. The Committee is strongly committed to returning inflation to its 2 percent objective.
            In assessing the appropriate stance of monetary policy, the Committee will continue to monitor the implications of incoming information for the economic outlook. The Committee would be prepared to adjust the stance of monetary policy as appropriate if risks emerge that could impede the attainment of the Committee's goals. The Committee's assessments will take into account a wide range of information, including readings on labor market conditions, inflation pressures and inflation expectations, and financial and international developments.
            Voting for the monetary policy action were Jerome H. Powell, Chair; John C. Williams, Vice Chair; Michael S. Barr; Michelle W. Bowman; Lisa D. Cook; Austan D. Goolsbee; Patrick Harker; Philip N. Jefferson; Neel Kashkari; Adriana D. Kugler; Lorie K. Logan; and Christopher J. Waller.
            For media inquiries, please email
            [email protected]
            or call 202-452-2955.
            Implementation Note issued November 1, 2023
            Last Update:
                        November 01, 2023
    '''
    # User inputs FOMC statement
    Statement = st.text_area('Enter FOMC statement:', statement_sample)

    # Button to run sentiment analysis
    run_sentiment_button = st.button('Run Sentiment Analysis')

    # Variables to store sentiment results
    Sentiment_label = None
    Sentiment_score_LM = None
    Sentiment = None

    if run_sentiment_button:
        # Preprocess statement and run sentiment model
        Statement_cleaned = cleanStatement(Statement)
        Sentiment_label = get_sentiment_from_pretrained_model(Statement_cleaned)[0]
        Sentiment = 1 if Sentiment_label == 'positive' else -1 if Sentiment_label == 'negative' else 0
        Sentiment_score_LM = get_sentiment_score_LM(Statement_cleaned)
        if Sentiment_score_LM is not None:
            Sentiment_score_LM = float(Sentiment_score_LM)
        st.session_state.sentiment_done = True
        st.session_state.sentiment = Sentiment
        st.session_state.sentiment_score_lm = Sentiment_score_LM
        st.session_state.sentiment_label = Sentiment_label

    if st.session_state.sentiment_done and st.session_state.sentiment is not None:
        # Display sentiment analysis results
        st.write(f"Sentiment: {st.session_state.sentiment_label}, Sentiment Score (LM): {st.session_state.sentiment_score_lm}")
   
        st.header('Input the following key economic indicators')
        
        # Get user input for economic indicators
        PCE = st.text_input('Enter PCE:', '121.385')
        Consumer_Sentiment = st.text_input('Enter Consumer_Sentiment:', '97.52642')
        CPI = st.text_input('Enter CPI:', '307.619')
        ISM_PMI = st.text_input('Enter ISM PMI:', '50.0')
        Housing_starts = st.text_input('Enter Housing_starts:', '1498')
        Unemp_rate = st.text_input('Enter Unemp_rate:', '3.9')
        SPX = st.text_input('Enter SPX:', '4607.6')
        Ten_yr_try = st.text_input('Enter 10y_try:', '4.6')

        
        # Form to submit the full model
        with st.form(key='prediction_form'):
            submit_button = st.form_submit_button('Submit and Run Full Model')

            if submit_button:

                new_record = pd.DataFrame({
                    'PCE': [float(PCE)],
                    'Consumer_Sentiment': [float(Consumer_Sentiment)],
                    'CPI': [float(CPI)],
                    'ISM PMI': [float(ISM_PMI)],
                    'Housing_starts': [float(Housing_starts)],
                    'Unemp_rate': [float(Unemp_rate)],
                    'SPX': [float(SPX)],
                    '10y_try': [float(Ten_yr_try)],
                    'sentiment_score_LM': [float(st.session_state.sentiment_score_lm)],
                    'Sentiment': [int(st.session_state.sentiment)]
                })
                print(st.session_state.sentiment_score_lm, st.session_state.sentiment)
                # Append new record to historical data
                combined_data = pd.concat([historical_records, new_record], ignore_index=True)
                combined_data.to_csv('../data/preprocessed/test.csv')
                combined_data_with_features = feature_preprocess(combined_data, target, num_features, cat_features).drop(target, axis = 1)
                combined_data_with_features.to_csv('../data/preprocessed/test2.csv')
                # Select only the last row (new record with calculated features)
                final_input = combined_data_with_features[selected_features].iloc[-1,:]
                final_input_2d = np.array(final_input).reshape(1, -1)
            
                st.markdown("## Our prediction given the user inputs")

                prediction = latest_model.predict(final_input_2d)

                # Show the inputted values
                if prediction[0] == 1:
                    result = 'increase'
                elif prediction[0] == -1:
                    result = 'decrease' 
                elif prediction[0] == 0:
                    result = 'hold'

                st.write(f"Our prediction is that the Fed will **{result}** the Fed Rate in October.")


if __name__ == "__main__":
    main()















