import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from datetime import date, timedelta, datetime
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="Anomaly Explore",
    page_icon=":fire:",
    layout="wide",
)

with st.expander("File Upload", True):
    uploaded_file=st.file_uploader("Choose a (csv) file to upload",
    type='csv',)

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file).dropna()

with st.expander("show data?", True):
    st.dataframe(df)

# create a DF for numeric columns only
num_cols = []
for c in df.columns:
    try:
        t = pd.to_numeric(df[c])
        num_cols.append(c)
    except:
        pass
num_df = df[num_cols].copy().dropna()

#with st.expander("Select columns for anomaly detection (must be numeric)", False):
if 1:
    selected_columns = st.multiselect(
        "Select numeric columns to keep for anomaly detection:",
        options=num_cols,
        default=num_cols,
    )

#with st.expander("Detect anomalies", False):
run_anomalies_btn = st.button("run anomaly detection algo & show important features")

#if run_anomalies_btn:
if 1:
    df_filt_col = df[selected_columns].copy()
    model = IsolationForest(n_jobs=-1)
    model.fit(df_filt_col)

    scores = model.decision_function(df_filt_col)
    anomaly = model.predict(df_filt_col)
    #df["scores"] = scores
    #df["anomaly"] = anomaly
    
    df.insert(0, "scores", scores, True)
    df.insert(0, "anomaly", anomaly, True)

    print("get_var_imp_btn pressed")
    print(df)
    
    def fitRF(df):
        varsX = df[selected_columns]
        vary = df['scores']
        regr = RandomForestRegressor(random_state=0, n_jobs = -1)
        regr.fit(varsX, vary)
        return regr
    
    def featImp(mod):
        # return pd.Series(mod.feature_importances_, index=['x','y','z'])
        return pd.DataFrame({'feature': selected_columns, 'importance': mod.feature_importances_})
    
    def plotImp(impDF):
        # return impDF.plot(kind='barh')
        #return px.bar(impDF, x='feature', y='importance') #, orientation='h')
        return px.bar(impDF, y='feature', x='importance', orientation='h')
    
    regr = fitRF(df)
    regr_featImp = featImp(regr)
    plotFeatImp = plotImp(regr_featImp)
    
    st.plotly_chart(plotFeatImp)

st.download_button(
        "Download data with anomaly scores added",
        data=df.to_csv(index=False),
        file_name="anomaly_data.csv",
    )

