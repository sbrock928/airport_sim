import pandas as pd
import numpy as np
import pathlib
from app import app



def queryData():

    # database query step goes here, importing for CSV for demo purposes
    PATH = pathlib.Path(__file__).parent
    DATA_PATH = PATH.joinpath("../datasets").resolve()
    df_imp = pd.read_csv(DATA_PATH.joinpath("2018_scrub.csv"))
    df_imp = df_imp[(df_imp['DIVERTED'] == 0) & (df_imp['CANCELLED']==0)]

    df_imp['DEP_DELAY'].fillna(0, inplace=True)
    df_imp['ARR_DELAY'].fillna(0, inplace=True)
    df_imp['AIR_TIME'].dropna(inplace=True)

    df_imp[["CRS_DEP_TIME", "DEP_TIME", "WHEELS_OFF", "WHEELS_ON", "CRS_ARR_TIME", "ARR_TIME"]] = df_imp[
       ["CRS_DEP_TIME", "DEP_TIME", "WHEELS_OFF", "WHEELS_ON", "CRS_ARR_TIME", "ARR_TIME"]].astype('str')


    for i in ["CRS_DEP_TIME", "DEP_TIME", "WHEELS_OFF", "WHEELS_ON", "CRS_ARR_TIME", "ARR_TIME"]:
        df_imp[i] = df_imp[i].apply(lambda x: x.zfill(4))

    df_imp['DEP_TIME'] = pd.to_datetime(df_imp['DEP_TIME'], format = '%H%M')
    df_imp['ARR_TIME'] = pd.to_datetime(df_imp['ARR_TIME'], format = '%H%M')



    return df_imp



