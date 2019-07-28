#!/usr/bin/env python
# coding: utf-8

# Es recomendable agregar la extensión jupyter-navbar.  
# > git clone https://github.com/shoval/jupyter-navbar.git  
# > cd jupyter-navbar  
# > python setup.py

# # Notas
# * Las fechas en el dataset corresponden a cuando se obtuvieron los datos.
# * Las fechas en los ejemplos corresponden a la predicción Y
#   
#   
# * Jorquera al parecer utiliza como Evento Crítico la concentración horaria mayor a 82 ppb. (ver pag. 3 de FORECASTING OZONE DAILY MAXIMUM LEVELS AT SANTIAGO, CHILE), en nuestro caso, un THETA > 82.
# * O3 percentil 0.75  --> 92.0
# * O3 percentil 0.90  --> 111.9
#   
#   
# * Invierno: 01 de Mayo al 31 de Agosto.   05-01    -    08-31
# * Verano: 01 de Noviembre al 31 de Marzo. 11-01    -    03-31
# 
# * Las Condes:
#     * v2004 - v2013 (2014)
#     * mean  76.176741
#     * std   20.619374
#     * std/mean 0.2706780800717111
#     * q0.5  75.557200 -> 76
#     * q0.75 89.000000 -> 89
# 
# * Independencia:
#     * v2009 - v2017 (2018)
#     * mean  45.802717
#     * std   15.391089
#     * std/mean 0.3360300438072265
#     * q0.5  44.195900 -> 46
#     * q0.75 54.333300 -> 56
# 
# * Parque O'Higgins
#     * v2009 - v2017 (2018)
#     * mean  50.853193
#     * std   14.191407
#     * std/mean 0.2790661935426552
#     * q0.5  51.042400 -> 50
#     * q0.75 60.256450 -> 60

# In[ ]:


15.391089/45.802717


# # Imports

# In[ ]:


import numpy as np
from tensorflow import set_random_seed
np.random.seed(123)
set_random_seed(2)

import os
import sys
import math
import pandas as pd
import matplotlib.pyplot as plt
import plotly.plotly as py
from plotly.offline import init_notebook_mode, enable_mpl_offline, iplot_mpl, iplot
import cufflinks as cf
from datetime import datetime

from ipywidgets import widgets, interactive, interact
from IPython.display import Javascript, display

from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score

from keras.models import Sequential
from keras.layers import Input, Dense, LSTM, Dropout, TimeDistributed
from keras.models import load_model
from keras.callbacks import EarlyStopping
import keras.backend as K

from hashlib import md5

#import utils
#from utils import gpr_invierno, gpr_verano



#init_notebook_mode(connected=True)
#cf.go_offline(connected=True)
#enable_mpl_offline()
GRAPH_IS_SET = False

ROOT = "./"
MODELS_FOLDER = "models"

try:
    os.mkdir(MODELS_FOLDER)
except:
    pass


# # Funciones Generales

# ## get_station

# In[ ]:


def get_station(STATION):
    if STATION == "Las_Condes":
        FILTER_YEARS = [2004, 2013]
        DEFAULT_THETA = 89
    elif STATION == "Independencia" or STATION == "Independencia2019":
        FILTER_YEARS = [2010, 2017]
        DEFAULT_THETA = 54
    elif STATION == "Parque_OHiggins" or STATION == "POH_full":
        FILTER_YEARS = [2009, 2016]
        DEFAULT_THETA = 60
    
    elif STATION == "CONDES 4F":
        FILTER_YEARS = [2008, 2013]
        DEFAULT_THETA = 84
    elif STATION == "INDEP 4F":
        FILTER_YEARS = [2009, 2014]
        DEFAULT_THETA = 58
    elif STATION == "POH 4F":
        FILTER_YEARS = [2009, 2014]
        DEFAULT_THETA = 60
    
    return STATION, FILTER_YEARS, DEFAULT_THETA
        


# ## for_graph

# In[ ]:


def for_graph():
    if "-f" not in sys.argv:
        return False
    global GRAPH_IS_SET
    if GRAPH_IS_SET == False:
        init_notebook_mode(connected=True)
        cf.go_offline(connected=True)
        enable_mpl_offline()
        GRAPH_IS_SET = True
    return True


# ## import_dataset

# In[ ]:


def import_dataset(STATION):
    if STATION == "Las_Condes" or STATION == "CONDES 4F":
        CSV = "data/dump-Las_Condes_2018-04-12_230000-verano.csv"
    
    elif STATION == "Independencia":
        CSV = "data/dump-Independencia_2018-04-12_230000-verano.csv"
    elif STATION == "Independencia2019":
        CSV = "data/dump-Independencia_2019-06-14_230000-verano.csv"
    
    elif STATION == "Parque_OHiggins":
        CSV = "data/dump-Parque_OHiggins_2019-06-12_230000-verano.csv"
    elif STATION == "POH_full":
        CSV = "data/dump-Parque_OHiggins_2019-06-14_230000-verano.csv"
    
    
    elif STATION == "INDEP 4F":
        CSV = "data/dump-Independencia_2019-06-14_230000-verano.csv"
    elif STATION == "POH 4F":
        CSV = "data/dump-Parque_OHiggins_2019-06-14_230000-verano.csv"
        
    DS = pd.read_csv(ROOT+CSV)
    #DS = pd.read_csv(ROOT+"data/dump-Independencia_2018-04-12_230000-verano.csv")
    DS["registered_on"] = pd.to_datetime(DS["registered_on"])
    DS.set_index("registered_on",inplace=True)
    return DS


# ## HourMax
# Para obtener la hora a la que ocure el máximo de un atributo en cada día.

# In[ ]:


def HourMax(feature,DF):
    maxf = "hm"+feature
    
    fecha = DF.index.hour
    fecha.tolist()
    DF[maxf] = fecha.tolist()
    
    g1 = DF.groupby(pd.Grouper(freq="D"))
    
    hmaxdiaria = []
    
    for fecha, group in g1:
        group = group[[feature,maxf]].dropna()
        if not group.empty:
            ar = group.values.tolist()
            m = max(ar)
            hourmax = m[1]
            hmaxdiaria.append(hourmax)
        else:
            hmaxdiaria.append(np.nan)

    return np.array(hmaxdiaria)


# ## precalcular_agregados
# Precalcula los agregados de los datos que se tienen.  
# Esto tarda por lo que el resultado es guardado en un archivo.  
# Se ejecuta la función inmediatamente para asi tener la variable PRECALC disponible como parametro por defecto para cuando se necesite.  

# In[ ]:


def precalcular_agregados(STATION,REMAKE=False):
    try:
        if REMAKE == False:
            agdf = pd.read_csv(ROOT+"precalcs/precalc_agregados_%s.csv"%STATION)
            agdf = agdf[ agdf.columns[1:] ]
        else:
            pd.read_csv("nunca_nadie_me_encontrara.nunca_jamas")
    except:
        df = import_dataset(STATION)
        agdf = pd.DataFrame()
        
        agregados = []
        
        # Calculo de la hora maxima de atributos en un día
        for feat in df.columns:
            maxf = "hm"+feat
            print("    For: ",feat)
            hmaxdiaria = HourMax(feat, df)
            print("        len(%s), "%maxf, len(hmaxdiaria))
            #agregados.append( (maxf, hmaxdiaria) )
            agdf[maxf] = hmaxdiaria
        
        agdf.to_csv(ROOT+"precalcs/precalc_agregados_%s.csv"%STATION)
        
    
    return agdf

PRECALC = precalcular_agregados("Las_Condes")


# ## precalcular_eventos
# Precalcula los eventos criticos del ozono.  
# Un evento crítico es cuando el promedio del ozono en una ventana de 8 horas supera un valor (theta) determinado.
# Entrega la cantidad de eventos criticos en un día y en otra columna si ocurrió un evento o no.

# In[ ]:


def precalcular_eventos(theta,STATION,REMAKE=False):
    print("Precalculandos Eventos Criticos ...")
    try:
        if REMAKE == False:
            ecdf = pd.read_csv(ROOT+"precalcs/precalc_ECat%.2f_%s.csv"%(theta, STATION), index_col="registered_on", parse_dates=True)
            ecdf = ecdf.asfreq("D")
        else:
            pd.read_csv("nunca_nadie_me_encontrara.nunca_jamas")
    except:
        df = import_dataset(STATION)
        
        g1 = df.groupby(pd.Grouper(freq="D"))
    
        eventos = []
        dates = []
        for fecha, group in g1:
            #print("========")
            #print(group["O3"])
            group = group[["O3"]].dropna().asfreq("h")
            count = 0
            #print("--")
            
            if len(group) >= 8:
                ###print(group)
                for fecha2 in group.index[:-7]:
                    i,f = fecha2, fecha2 + np.timedelta64(7,"h")
                    gmean = group[i:f].mean()[0]
                    if gmean >= theta:
                        count += 1
                if count > 0:
                    ec = 1
                else:
                    ec = 0
                ###print(fecha, count, ec)
                eventos.append( [count, ec] )
            else:
                ###print(fecha, np.nan)
                eventos.append( [np.nan, np.nan])
            dates.append(fecha)
        eventos = np.array(eventos)
        ecdf = pd.DataFrame(eventos, columns=["countEC","EC"])
        ecdf["registered_on"] = np.array(dates)
        ecdf = ecdf.set_index("registered_on")
        #print(ecdf["EC"])
        
        ecdf.to_csv(ROOT+"precalcs/precalc_ECat%.2f_%s.csv"%(theta, STATION), index=True)

    return ecdf

asdf = precalcular_eventos(61,"Independencia",REMAKE=False)


# ## import_merge_and_scale
# Importa el dataset.  
# Agrega los datos AGREGADOS. Ej. la hora a la que ocurre el máximo de un atributo.  
# Y escala los datos.

# In[ ]:


def import_merge_and_scale(Config, verbose=True, SCALE = True):
    STATION = Config["STATION"]
    AGREGADOS = Config["AGREGADOS"]
    TARGET = Config["TARGET"]
    THETA = Config["THETA"]
    PRECALC = Config["PRECALC"]
    SHIFT = Config["SHIFT"]
    PAST = Config["PAST"]
    SCALER = Config["SCALER"]
    FILTER_YEARS = Config["FILTER_YEARS"]
    IMPUTATION = Config["IMPUTATION"]
    if "GRAPH" in Config:
        GRAPH = Config["GRAPH"]
        if GRAPH == True:
            for_graph()
    #PREDICTED_CE = Config["PREDICTED_CE"]
    
    
    #SCALE = True
    
    vprint = print if verbose else lambda *a, **k: None
    
    
    print("Importing Dataset and Scaling...")
    DS = import_dataset(STATION)
    vprint("    Dataset imported.")
    
    vprint("    Adjuntando Agregados Precalculados.")
    agregados = []

    if AGREGADOS == ["ALL"]:
        AGREGADOS = DS.columns
    
    for feat in AGREGADOS:
        maxf = "hm"+feat
        vprint("        Adding ",maxf)
        hmaxdiaria = PRECALC[maxf]
        agregados.append( (maxf, hmaxdiaria) )
    
    
    
    # Agrupar por maximos diarios
    gp = DS.groupby(pd.Grouper(freq='D'))
    df = gp.aggregate(np.max)
    #df = DS#.asfreq("H")
    
    #TARGET_MASK = np.isnan(df[TARGET]) != True
    
    if IMPUTATION:
        if IMPUTATION != "mean":
            df = df.fillna(method=IMPUTATION)
        else:
            df = df.fillna(df.mean())
    
    for name, data in agregados:
        df[name] = data.values
    
    # Agregando eventos criticos
    ecdf = precalcular_eventos(THETA, STATION, REMAKE=False)
    df = pd.concat([df, ecdf],axis=1)
    
    
    
    # Agregar si O3 > THETA
    df["O3btTHETA"] = (df["O3"].dropna() >= THETA)*1.
    
    #print(df)
    
    
    #Scaler - normalize
    if SCALE:
        #min_max_scaler = preprocessing.MinMaxScaler()
        #miScaler = preprocessing.StandardScaler()
        miScaler = SCALER()
        norm_data = pd.DataFrame(miScaler.fit_transform(df), columns=df.columns, index=df.index)
        #norm_data["EC"] = ecdf["EC"].values
        vprint("    Dataset Scaled.")
    else:
        #ONLY FOR TEST
        norm_data = df
        for i in range(0,5):
            print("")
        print("    ===========================================================")
        print("    ==================DATASET SIN ESCALAR - OJO================")
        print("    ===========================================================")
    
    #Scaler para el Y
    #Yscaler = preprocessing.MinMaxScaler()
    #Yscaler = preprocessing.StandardScaler()
    Yscaler = SCALER()
    Yscaler.fit(df[TARGET].values.reshape((-1,1)))
    
    #print(norm_data["y"])
    if TARGET in ["EC","O3btTHETA"]: #Feature booleana
        norm_data["y"] = df[TARGET].values.reshape((-1,1))
    else:
        norm_data["y"] = Yscaler.transform( df[TARGET].values.reshape((-1,1)) )
    #print(df)

    #Scaler para datos horarios
    #h24scaler = preprocessing.MinMaxScaler()
    #h24scaler = preprocessing.StandardScaler()
    h24scaler = SCALER()
    ###h24scaler.fit(np.array(range(0,24),dtype="float64")[:, None])
    
    # Adding agregados escalados
    ###for name, data in agregados:
    ###    norm_data[name] = h24scaler.transform(data[:,None])
    ###vprint("    Agregados escalados.")
    
    # Adding Target
    YLABELS = []
    if SHIFT >= 0:
        # Predecir el TARGET actual o en el pasado.
        #norm_data["y"] = norm_data["TARGET"].shift(SHIFT)
        #TARGET_TO_SHIFT = TARGET_TO_SHIFT
        norm_data["y"] = norm_data["y"].shift(SHIFT)
        YLABELS.append("y")
        norm_data = norm_data.drop(TARGET, axis=1)
        print("    SHIFT==0        COLUMN %s removed from features"%TARGET)
        vprint("    y Target added.")
    #elif SHIFT == -1:
    #    # PREDECIR EL SIGUIENTE DÍA
    #    norm_data["y"] = norm_data[TARGET].shift(SHIFT)
    #    YLABELS.append("y")
    #    vprint("    y Target added.")
    else: #SHIFT < -1
        # PREDECIR VARIOS DIAS HACIA ADELANTE
        #norm_data["y"] = norm_data[TARGET].shift(-1)
        norm_data["y"] = norm_data["y"].shift(-1)
        YLABELS.append("y")
        vprint("    y Target added.")
        M = SHIFT * -1
        if PAST is False:
            c = 1
        else:
            c = PAST*-1
        while c <= M:
            if c == 1:
                c += 1
                continue
            #norm_data["y"+str(c)] = norm_data[TARGET].shift(-1*c)
            #TARGET_MASK = TARGET_MASK.shift(-1)
            norm_data["y"+str(c)] = norm_data["y"].shift(-1*c)
            #norm_data["y"] = norm_data["y"].where(TARGET_MASK)
            YLABELS.append("y"+str(c))
            vprint("    y%d Target added."%(c))
            c += 1
    #print(norm_data["y"].copy())
    #return
    
    #norm_data = norm_data.fillna(method="ffill")
    
    if FILTER_YEARS:
        i,f = FILTER_YEARS
        norm_data = norm_data["%d-11-01"%i:"%d-03-31"%(f+1)]
        #print(norm_data)
    dataset = norm_data
    
    dataset = dataset[ (dataset.index.month>=11) |  (dataset.index.month<=3) ]
    
    #dataset["O3"] = dataset["O3"]["2004-11-01":"2004-11-05"]
    
    
    #print(dataset)
    
    return dataset, YLABELS, Yscaler, h24scaler


# In[ ]:


# Alias for test
#STATION, FILTER_YEARS, THETA = "POH_full", [2010, 2017], 56
#STATION, FILTER_YEARS, THETA = "Las_Condes", [2004, 2013], 89
#STATION, FILTER_YEARS, THETA = "Parque_OHiggins", [2009, 2017], 60
STATION, FILTER_YEARS, THETA = get_station("INDEP 4F")
tempConfig = {
    "STATION": STATION,
    "SCALER" : preprocessing.StandardScaler,
    "AGREGADOS": [],#["ALL"],
    "PRECALC": [], #precalcular_agregados(STATION),
    "IMPUTATION": None,
    "THETA": THETA,
    "TARGET":"O3",
    "SHIFT":-1,
    "PAST":False,
    "FILTER_YEARS": FILTER_YEARS,
}
norm_data, YLABELS, dataScaler, __scaler = import_merge_and_scale(tempConfig, SCALE = False)
original = norm_data


# In[ ]:


from utils import gpr_invierno, gpr_verano
gp = ((original["O3"] )*1).groupby(gpr_verano)
print(gp.aggregate(np.mean))

gp = ((original["O3"] > 82 )*1).groupby(gpr_verano)
print(gp.aggregate(np.sum))


# In[ ]:


print(len(original["O3"]))
original["O3"].dropna()
mask = np.isnan(original["O3"])
original["O3"][mask]
original["O3"].quantile(0.75)


# In[ ]:


#STOP


# In[ ]:


a = pd.DataFrame([1,2,3, float("nan")])
n = preprocessing.StandardScaler()
n.fit(a)
n.transform(a)
m = a.where( np.isnan(a) != True)
m

b = pd.DataFrame([1,2,3,4])
b.where( np.isnan(a) != True )


# ## select_features
# Crea un DataFrame que contiene sólo los atributos indicados en FEATURES o los datos que contengan un porcentaje de no nulos mayor al CUT. Cuando se utiliza el CUT, se pueden banear atributos manualmente.  
# El DataFrame devuelto contiene todas filas en donde ningún dato es no nulo.
#Nota:  CUT    DIAS_DISPONIBLES         PREDICTORES
#      0.40,         3004             ['CO' 'PM10' 'O3']
#      0.30,         2046             ['CO' 'PM10' 'PM25' 'NO' 'NOX' 'UVA' 'UVB' 'O3']
#      0.28,         1712             ['CO' 'PM10' 'PM25' 'NO' 'NOX' 'RH' 'TEMP' 'UVA' 'UVB' 'O3']
#      0.25,         1629             ['CO' 'PM10' 'PM25' 'NO' 'NOX' 'WD' 'RH' 'TEMP' 'WS' 'UVA' 'UVB' 'O3']
# In[ ]:


# Selección de atributos en base a la cantidad de ejemplos sin datos nulos que se dispondrán
def select_features(internalConfig, Config, verbose=True):
    dataset = internalConfig['complete_dataset']
    ylabels = internalConfig['ylabels']
    FEATURES = Config['FIXED_FEATURES']
    CUT = Config['CUT']
    BAN = Config['BAN']
    SHIFT = Config['SHIFT']
    
    vprint = print if verbose else lambda *a, **k: None
    
    print("Selecting features ...")
    if FEATURES == []:
        vprint("    Using CUT=%.2f"%CUT)
        
        a=dataset.isna().sum()
        b=dataset.describe().iloc[0]
        cantidad = (b/(a+b))
        
        atributos = cantidad[cantidad >= CUT]
        excluidos = cantidad[cantidad < CUT]

        index = atributos.index.values.tolist()
        banned = []
        for b in BAN:
            if b in index:
                index.remove(b)
                banned.append(b)
            if "hm"+b in index:
                index.remove("hm"+b)
                banned.append("hm"+b)
        
        vprint("    %i Atributos Excluidos:"%(len(excluidos)), excluidos.index.values)
        vprint("    %i Atributos Baneados:"%(len(banned)),banned)
    else:
        vprint("    Using fixed features ...")
        if "y" in FEATURES:
            print("    WARNING: 'y' no debe estar en los ATRIBUTOS, se removerá y agregara al final del dataset")
            FEATURES.remove("y")
            
            
        index = FEATURES + ylabels


    

    data = dataset[index]
    
    #print(data.dropna(how="all"))
    #temp = data.dropna(how="all")
    #temp = temp.drop("y",axis=1)
    #temp.to_csv("data_con_nan.csv",na_rep="NaN")
    
    
    #Removiendo ejemplos con al menos un atributo NaN
    data.dropna(inplace=True)
    
    FEATURES = index
    for y in ylabels:
        FEATURES.remove(y)
    vprint("    %i Atributos Seleccionadios:"%(len(FEATURES)),FEATURES)
    vprint("    Cantidad de dias totales disponibles:",len(data))

    
    
    return data, FEATURES


# In[ ]:


#STATION, FILTER_YEARS, THETA = "Independencia", [2009, 2017], 56
#STATION, FILTER_YEARS, THETA = "Las_Condes", [2004, 2013], 92
STATION, FILTER_YEARS, THETA = "Parque_OHiggins", [2009, 2017], 60
tempConfig = {
        "STATION":STATION,
        "SCALER" : preprocessing.StandardScaler,
        "IMPUTATION" : None,# "ffill",
        "AGREGADOS":[],
        "PRECALC":precalcular_agregados(STATION),
        "THETA":THETA,
        "TARGET":"O3",
        "SHIFT":-1,
        "PAST":False,
        "FIXED_FEATURES":['CO', 'PM10', 'PM25', 'NO', 'NOX', 'WD', 'RH', 'TEMP', 'WS', 'UVA', 'UVB', 'O3'],
        "CUT": 0.41, #0.26 para 12 atributos con todos los años
        "BAN": ["countEC","EC", "O3btTHETA"],
        "FILTER_YEARS" : FILTER_YEARS
    }
tempIC={}
tempIC["complete_dataset"], tempIC["ylabels"], __Yscaler, __h24scaler = import_merge_and_scale(tempConfig, verbose=False, SCALE=True)
data, __ = select_features(tempIC, tempConfig, verbose=True)


# In[ ]:


data


# In[ ]:


#kk = pd.read_csv("data_con_nan-CUT=0.26.csv", index_col=0, parse_dates=True)


# In[ ]:


#kk = data.drop("y",axis=1).dropna(how="all")
#pd.read_csv("data_horaria_con_nan_CUT=0.55.csv", index_col=0, parse_dates=True)


# ## nonancheck
# Función de control para asegurarse que no existan datos no nulos.  
# La funcion es estatica por lo que los valores deben cambiarse dentro de la función si es que se desea probar algo más.

# In[ ]:


#Asegurandose que no hayan datos faltantes
def nonancheck():
    STATION = "Las_Condes"
    tempConfig = {
        "STATION" : STATION,
        "SCALER" : preprocessing.StandardScaler,
        "IMPUTATION":None,
        "FILTER_YEARS" : [],
        "AGREGADOS":[], "PRECALC":precalcular_agregados(STATION),
        "THETA":61,
        "TARGET":"O3",
        "SHIFT":-1,
        "PAST":False,
        "FIXED_FEATURES":[],
        "CUT": 0.28,
        "BAN": [],
        
    }
    tempIC={}
    tempIC["complete_dataset"], tempIC["ylabels"], __Yscaler, __h24scaler = import_merge_and_scale(tempConfig, verbose=False)
    data, __ = select_features(tempIC, tempConfig, verbose=False)
    return data.isna().describe()

nonancheck()


# In[ ]:


#PRUEBAS - CONTROL DE DATOS
def y_vs_target():
    STATION = "Las_Condes"
    tempConfig = {
        "STATION" : STATION,
        "SCALER" : preprocessing.StandardScaler,
        "IMPUTATION":None,
        "FILTER_YEARS" : [],
        "AGREGADOS":[], "PRECALC":precalcular_agregados(STATION),
        "THETA":61,
        "TARGET":"O3",
        "SHIFT":-1,
        "PAST":False,
        "FIXED_FEATURES":[],
        "CUT": 0.28,
        "BAN": [],
        "GRAPH":True,
    }
    tempIC={}
    tempIC["complete_dataset"], tempIC["ylabels"], __Yscaler, __h24scaler = import_merge_and_scale(tempConfig, verbose=False)
    data, __ = select_features(tempIC, tempConfig, verbose=False)
    
    dataset = tempIC["complete_dataset"]
    TARGET = tempConfig["TARGET"]

    oo = data[TARGET]
    yy = dataset["y"]

    gg = pd.concat([yy,oo], axis=1)

    gg.iplot()

if for_graph():
    y_vs_target()


# ## obtener_secuencias
# Obtiene las secuencias de distintos largos máximos posibles.  
# Devuelve un diccionario donde la llave es el largo de las secuencia y el valor es una lista con tuplas con la fecha inicial y final correspondiente a dicha secuencia de ese largo.  
# Ej. { 1 : [ ('2000-01-01', '2000-01-01') ], 3 : [ ('2006-01-01', '2006-01-03'), ('2006-02-03', '2006-02-05') ]}

# In[ ]:


#Obtenemos la cantidad se secuencias de distinto largo maximo que se pueden hacer con los días correlativos.
def obtener_secuencias(internalConfig):
    data = internalConfig["data"]
    
    secuencias = {}
    fecha = data.index[0] 
    fin = data.index[-1] + np.timedelta64(1,"D")
    largo = 0
    i = 0
    sec_i = fecha
    while True:
        i += 1
        siguiente = fecha + np.timedelta64(1,"D")
        if siguiente in data.index:
            fecha = siguiente
        else:
            if i not in secuencias:
                secuencias[i] = []
            secuencias[i].append( (sec_i,fecha) )
            i = 0
            f_index = data.index.get_loc(fecha) + 1
            if f_index != len(data.index):
                fecha = data.index[f_index]
                sec_i = fecha
            else:
                break
    return secuencias
        
#SECUENCIAS = secuencias


# ## make_examples
# Recibe el dataset, las secuencias, la cantidad de TIMESTEP y crea los ejemplos con su correspondiente etiqueta.
# 
# Por ejemplo, dado un TIMESTEP de 3.  
# Cada ejemplo tendrá 3 días sucesivos y una etiqueta Y asociada al ozono del día siguiente.  
#     Es decir, se utilizaran los datos en el tiempo T, T-1 y T-2 para predecir el ozono en tiempo T+1:  
#     (T-2),(T-1),(T)     ->   (T+1)  
# 
# Desde el punto de vista del dataset, corresponde a 3 filas sucesivas en orden cronológico donde la etiqueta corresponde al valor Y de la última fila.
# 
# La función devuelve:  
# > Un array 'examples' con los ejemplos de entrenamiento con el shape (cantidad, TIMESTEP, FEATURES).  
# > Un array 'y' con los Y en un arreglo bidimensional de sólo una columna con el shape (cantidad, 1).  
# > Un array 'date_index' con las fechas asociadas la PREDICCIÓN Y.  
# 
# Los tres array estan ordenados temporalmente por lo que la date_index[i] corresponde a la etiqueta y[i] de los ejemplos examples[i]

# In[ ]:


#Haremos los distintos ejemplos segun el TIMESTEP indicado.
#NOTA: La fecha adjunta en cada ejemplo y lista 'y' corresponde a la fecha de la predicción 'y'
#      y no a fecha de los datos, esto para poder graficar de forma ordenada despúes.
#      Es decir, para predecir el ozono en un día X, tomo los datos de los TIMESTEP dias anteriores.
def make_examples(internalConfig, Config, verbose=True):
    data = internalConfig["data"]
    secuencias = internalConfig["secuencias"]
    ylabels = internalConfig["ylabels"]
    TIMESTEP = Config["TIMESTEP"]
    OVERLAP = Config["OVERLAP"]
    
    
    vprint = print if verbose else lambda *a, **k: None
    
    print("Making examples ...")
    #TIMESTEP= 3
    #OVERLAP = True
    
    largos = list(secuencias.keys())
    largos.sort()
    examples = []
    y = []
    ylen = len(ylabels)
    
    for l in largos:
        #print(l)
        if l < TIMESTEP:
            continue
        for sec in secuencias[l]:
            inicio, fin = sec
            #print(sec)
            s = data[inicio:fin].values
            if OVERLAP:
                i = 0
                while i <= len(s) - TIMESTEP:
                    c = 0
                    new_example = []
                    yy = []
                    while c < TIMESTEP:
                        new_example.append( s[i+c][:-ylen] )
                        #TimeDistributed
                        yy.append( s[i+c][-ylen:] )
                        c+=1
                    fecha = inicio + np.timedelta64(i+c, "D")
                    ##new_example.reverse()
                    examples.append( [fecha] + new_example  )
                    ##y.append( ( fecha, s[i+c-1][-1] ) )
                    ##y.append( [ fecha] + s[i+c-1][-ylen:].tolist() )
                    #TimeDistributed
                    y.append(( [fecha] + yy ))
                    i += 1
            else:
                i = 0
                while i <= len(s) - TIMESTEP:
                    c = 0
                    new_example = []
                    #yy=[]
                    while c < TIMESTEP:
                        new_example.append( s[i][:-1] )
                        #yy.append( s[i][-1] )
                        c += 1
                        i += 1
                    fecha = inicio + np.timedelta64(i, "D")
                    ##new_example.reverse()
                    examples.append( [fecha] + new_example  )
                    y.append( ( fecha, s[i-1][-1] ) )
                    #y.append(( [fecha] + yy ))
            #break
        #break

        
    print("    Ejemplos Disponibles: , ",len(examples))
    vprint("    len(y), ",len(y))

    
    # Sort by Date of prediction Y
    examples.sort()
    y.sort()
    
   
    examples = np.array(examples)
    y = np.array(y)
    
    # Get date index of all examples in order. To use it later
    date_index = y[:,0]
    vprint("    len(date_index), ", len(date_index))
    
    
    examples = np.array( examples[:,1:].tolist() )
    y = np.array( y[:,1:].tolist() )
    #y = y[:,1:]
    
    vprint("    examples.shape :", examples.shape )
    vprint("    y.shape :", y.shape  )
    return examples, y, date_index


# ## join_index
# Función auxiliar que se utiliza para indexar un columna de fechas a una columna de datos, y así poder asociarlas temporalmente junto con otras columnas o datasets.

# In[ ]:


# Une la fecha a un arreglo de valores, ambos de igual largo
# Retorna un DataFrame
def join_index(date_index, array, label):
    date_index = date_index.reshape( (-1,1) )
    array = array.reshape( (-1,1) )
    df = pd.DataFrame(np.hstack([date_index, array]))
    df.columns = ["fecha",label]
    df = df.set_index("fecha")
    return df


# ## plot_y_true
# Función estática y de control.  
# Para todos los ejemplos, grafica el valor de Y y la característica que representa en los distintos TIMESTEP.  
# Para el ejemplo, grafica el Ozono como TARGET con un TIMESTEP de 3

# In[ ]:


# Plot del TARGET y su valor actual
def plot_y_true():
    #STATION = "Las_Condes"
    STATION, FILTER_YEARS, THETA = get_station("INDEP 4F")
    tempConfig = {
        "STATION":STATION,
        "SCALER" : preprocessing.StandardScaler,
        "IMPUTATION":None,
        "FILTER_YEARS" : FILTER_YEARS,
        "AGREGADOS":[],
        "PRECALC":precalcular_agregados(STATION),
        "THETA":THETA,
        "TARGET":"O3",
        "TIMESTEP":5,
        "OVERLAP":True,
        "SHIFT":-1,
        "PAST":False,
        "FIXED_FEATURES":['CO', 'PM10', 'PM25', 'NO', 'NOX', 'WD', 'RH', 'TEMP', 'WS', 'UVA', 'UVB', 'O3'],
        "CUT": 0.41,
        "BAN": ["EC","countEC", 'O3btTHETA'],
        "GRAPH": True
    }
    tempIC={}
    tempIC["complete_dataset"], tempIC["ylabels"], __Yscaler, __h24scaler = import_merge_and_scale(tempConfig, verbose=False, SCALE=False)
    tempIC["data"], tempIC["features"] = select_features(tempIC, tempConfig, verbose=True)
    tempIC["secuencias"] = obtener_secuencias(tempIC)
    examples, y, date_index = make_examples(tempIC, tempConfig, verbose=False)
    
    FEATURES = tempIC["features"]
    TARGET = tempConfig["TARGET"]
    TIMESTEP = tempConfig["TIMESTEP"]
    
    f = FEATURES.index(TARGET)
    
    print(y.shape)

    df = join_index(date_index, y[:,-1,0], "y (T+1)")
    ### El indice esta mostrando el ozono, es un array, no un dataframe
    
    for i in range(0, TIMESTEP):
        label = "T" if i==0 else "T-%d"%i
        gf = join_index(date_index, examples[:,TIMESTEP-1-i,f], label)
        df = pd.concat([df,gf], axis=1)
    df.asfreq("D").iplot(title="TARGET: %s, index: %d"%(TARGET,f), hline=[THETA])

    #df[np.datetime64("2014-01-03"):np.datetime64("2014-01-08")]
    
    
if for_graph():
    plot_y_true()


# ## make_traintest
# Divide los array 'examples', 'y' y 'data_index', según un porcentaje dado como trainset y testset.  
# Además recorta el trainset y testset para hacerlos divisibles por el BATCH_SIZE.

# In[ ]:


#Split train and test datasets
def make_traintest(internalConfig, Config, verbose=True):
    examples = internalConfig["examples"]
    y = internalConfig["y_examples"]
    date_index = internalConfig["dateExamples"]
    y_len = internalConfig["y_len"]
    BATCH_SIZE = Config["BATCH_SIZE"]
    TRAINPCT = Config["TRAINPCT"]
    TIMEDIST = Config["TIMEDIST"]
    TIMESTEP = Config["TIMESTEP"]
    SHUFFLE = Config["SHUFFLE"]
    
    
    vprint = print if verbose else lambda *a, **k: None
    
    print("Making Trainset y Testset ...")
    train_size = int( len(examples)*TRAINPCT )
    test_size = len(examples) - train_size
    
    ### shuffle examples
    if SHUFFLE:
        np.random.seed(123)
        shuffle_index = list(range(len(examples)))
        np.random.shuffle(shuffle_index)
        examples = examples[shuffle_index]
        date_index = date_index[shuffle_index]
        y = y[shuffle_index]
    
    np.random.seed(123)  
    
    
    trainX, trainY = examples[0:train_size], y[0:train_size]
    dateTrain = date_index[0:train_size]
    
    testX, testY = examples[train_size:], y[train_size:]
    dateTest = date_index[train_size:]
    
    
    ###Cortar los dataset para hacerlos calzar con el BATCH_SIZE, necesario cuando se hace stateful
    ##vprint("    len of Trainset before clip: ", len(trainX))
    ##vprint("    len of Testset before clip: ", len(testX))
    ##
    ##cuttr = int(train_size/BATCH_SIZE)*BATCH_SIZE
    ##trainX = trainX[:cuttr]
    ##trainY = trainY[:cuttr]
    ##dateTrain = dateTrain[:cuttr]
    ##print("        Discargind %d last examples in Trainset to match with batch size of %d"%(train_size-cuttr, BATCH_SIZE))
    ##
    ##cuttst = int(test_size/BATCH_SIZE)*BATCH_SIZE
    ##testX = testX[:cuttst]
    ##testY = testY[:cuttst]
    ##dateTest = dateTest[:cuttst]
    ##print("        Discargind %d last examples in Testset to match with batch size of %d"%(test_size-cuttst, BATCH_SIZE))
    
    ###
    ### VALIDATION SPLIT
    ###
    train_size = int( len(trainX)*0.85 )
    validation_size = len(trainX) - train_size
    print(len(trainX),train_size,validation_size)
    
    tempX = trainX
    tempY = trainY
    tempDate = dateTrain
    
    trainX, trainY = tempX[0:train_size], tempY[0:train_size]
    dateTrain = tempDate[0:train_size]
    
    validX, validY = tempX[train_size:], tempY[train_size:]
    dateValid = tempDate[train_size:]
    
    #print("trainY.shape, ",trainY.shape)
    #print("validX.shape,", validX.shape)
    #print("validY.shape, ",validY.shape)
    
    if TIMEDIST == True:
        trainY = trainY.reshape(-1, TIMESTEP, y_len )
        validY = validY.reshape(-1, TIMESTEP, y_len )
        testY  = testY.reshape( -1, TIMESTEP, y_len )
    else:
        trainY = trainY[:,-1]
        validY = validY[:,-1]
        testY  = testY[:,-1]
    
    
    
    vprint("    trainX.shape, ", trainX.shape)
    vprint("    validX.shape, ", validX.shape)
    vprint("    testX.shape , ", testX.shape)
    
    vprint("    trainY.shape, ", trainY.shape)
    vprint("    validY.shape, ", validY.shape)
    vprint("    testY.shape, ", testY.shape)
    
    d = {
        'trainX':      trainX,
        'trainY':      trainY,
        'validX':      validX,
        'validY':      validY,
        'testX' :      testX,
        'testY' :      testY,
        'dateTrain'  : dateTrain,
        'dateValid'  : dateValid,
        'dateTest'   : dateTest
    }
    
    return d
    #return trainX, trainY, validX, validY, testX, testY, dateTrain, dateValid, dateTest


#join_index(dateTrain, trainY,"train").iplot()
#join_index(dateTest, testY,"test").iplot()


# ## make_folds_TVT

# In[ ]:


def make_folds_TVT(internalConfig, Config):
    examples = internalConfig["examples"]
    y_examples = internalConfig["y_examples"]
    dateExamples = internalConfig["dateExamples"]
    LAGS = Config["TIMESTEP"]
    STARTYEAR, ENDYEAR = Config["FILTER_YEARS"]
    
    list_trainX = []
    list_trainY = []
    list_validX = []
    list_validY = []
    list_testX = []
    list_testY = []
    list_dateTrain = []
    list_dateValid = []
    list_dateTest = []
    
    inicio = "{}-11-01".format
    fin = "{}-03-31".format
    dates = dateExamples.astype("datetime64")
    firstyear = int(str(dateExamples[0])[0:4])
    firstyear = STARTYEAR
    for year in range(firstyear, ENDYEAR-1):
        trainRange = [ np.datetime64(inicio(firstyear)), np.datetime64(fin(year+1)) ]
        validRange = [ np.datetime64(inicio(year+1))   , np.datetime64(fin(year+2)) ]
        testRange =  [ np.datetime64(inicio(year+2))   , np.datetime64(fin(year+3)) ]
        #print(year)
        #print(trainRange)
        #print(validRange)
        #print(testRange)
        
        mask = (trainRange[0] <= dates) & (dates <= trainRange[1])
        trainX = examples[mask]
        trainY = y_examples[mask][:,-1]
        dateTrain = dateExamples[mask]
        #print(mask)
        
        mask = (validRange[0] <= dates) & (dates <= validRange[1])
        validX = examples[mask]
        validY = y_examples[mask][:,-1]
        dateValid = dateExamples[mask]
        #print(mask)
        
        mask = (testRange[0] <= dates) & (dates <= testRange[1])
        testX = examples[mask]
        testY = y_examples[mask][:,-1]
        dateTest = dateExamples[mask]
        #print(mask)
        
        if len(trainX) != 0 and len(validX) != 0 and len(testX) != 0:
            list_trainX.append(trainX)
            list_trainY.append(trainY)
            list_validX.append(validX)
            list_validY.append(validY)
            list_testX.append(testX)
            list_testY.append(testY)
            list_dateTrain.append(dateTrain)
            list_dateValid.append(dateValid)
            list_dateTest.append(dateTest)
            print("Usando {}-{}, lags: {}".format(firstyear,year+2,LAGS), (len(trainX),len(validX),len(testX)) )
        else:
            print("{}-{} excluido, lags: {}, examples:".format(firstyear,year+2,LAGS), (len(trainX),len(validX),len(testX)) )
    
    d = {
        'list_trainX':list_trainX,
        'list_trainY':list_trainY,
        'list_validX':list_validX,
        'list_validY':list_validY,
        'list_testX' :list_testX,
        'list_testY' :list_testY,
        'list_dateTrain'  :list_dateTrain,
        'list_dateValid'  :list_dateValid,
        'list_dateTest'   :list_dateTest
    }
    
    return d


# ## create_filename

# In[ ]:


def create_filename(internalConfig, Config):
    global MODELS_FOLDER
    if "subFeatures" in Config:
        features = Config["subFeatures"]
        f_len = len(features)
    else:
        features = internalConfig["features"]
        f_len = internalConfig["f_len"]
    THETA = Config["THETA"]
    FUTURE = Config["FUTURE"]
    PAST = Config["PAST"]
    TARGET = Config["TARGET"]
    BATCH_SIZE = Config["BATCH_SIZE"]
    STATION = Config["STATION"]
    SEED = Config["SEED"]
    if "subTimeStep" in Config:
        TIMESTEP = Config["subTimeStep"]
    else:
        TIMESTEP = Config["TIMESTEP"]
    TRAINPCT = Config["TRAINPCT"]
    #OVERWRITE_MODEL = Config["OVERWRITE_MODEL"]
    MODEL_NAME = Config["MODEL_NAME"]
    LAYERS = Config["LAYERS"]
    EPOCHS = Config["EPOCHS"]
    DROP_RATE = Config["DROP_RATE"]
    TIMEDIST = Config["TIMEDIST"]
    SHUFFLE = Config["SHUFFLE"]
    if "QUANTILES" in Config:
        strQUANTILES = "-" + str(Config["QUANTILES"]).replace("0.",".")
    else:
        strQUANTILES = ""
        
    if "isLSTM" in Config:
        isLSTM = "-"+str(Config["isLSTM"])
    else:
        isLSTM = ""
    
    if "qfileHash" in internalConfig:
        qfileHash = "-"+str(internalConfig["qfileHash"])
    else:
        qfileHash = ""
    if "LOSS" in Config:
        MODEL_NAME += Config["LOSS"]
    
    if Config["FOLDS_TVT"]:
        foldmax = len(internalConfig["list_trainX"])
        currentFold = internalConfig["fold"]
        TVT = "-TVT%sof%s"%(currentFold,foldmax)
    else:
        TVT = ""
    
    if Config["FILTER_YEARS"]:
        i,f = Config["FILTER_YEARS"]
        fyears = "-%d-%d"%(i,f)
    else:
        fyears = ""
        
    IMPUT = Config["IMPUTATION"]
    
    strFEATURES = str(features).replace("'","")
    strTD = "TD"*TIMEDIST
    FILE_NAME = "./%s/%s-%.2f-(%d,%d,%d)%s-%s-%.3f-%s-%s-%d-(%s,%s)-%s-%s%sSHU%s%s%s%s%s%s%s.h5"%(MODELS_FOLDER, MODEL_NAME, THETA, BATCH_SIZE, TIMESTEP, f_len, strQUANTILES, strFEATURES, TRAINPCT, TARGET, LAYERS, EPOCHS, str(PAST), FUTURE, strTD, DROP_RATE, isLSTM , SHUFFLE,TVT, fyears,STATION,SEED,IMPUT, qfileHash )
    FILE_NAME = FILE_NAME.replace(" ","")
    
    return FILE_NAME


# # Metricas

# ## RMSE y MAE

# In[ ]:


def RMSE(ytrue, ypred, THETA=False, norm=False, ALL=False):
    ytrue = ytrue.reshape(-1)
    ypred = ypred.reshape(-1)
    if ALL == True:
        return {"RMSE"    : RMSE(ytrue, ypred),
                "RMSPE"   : RMSE(ytrue, ypred, norm=True),
                "RMSEat"  : RMSE(ytrue, ypred, THETA=THETA),
                "RMSPEat" : RMSE(ytrue, ypred, THETA=THETA, norm=True)
               }
    
    if THETA is False:
        includes = np.ones(len(ytrue), dtype=bool)
    else:
        includes = ytrue >= THETA
    
    if includes.any():
        yt = ytrue[includes]
        yp = ypred[includes]
        e = yt - yp
        if norm == True:
            e = e/yt
            #if THETA == False:
            #    #for i in range(len(e)):
            #    #    print(yt[i],yp[i],yt[i]-yp[i], np.abs(e[i]))
            #    print(np.mean(np.abs(e)))
        #print("###")
        e2 = e**2
        
        mean = np.mean(e2)
        #if THETA == False and norm == True:
        #    for i in range(len(e2)):
        #        if (np.abs(e[i]) >= 1):
        #            print(yt[i],yp[i],yt[i]-yp[i], np.abs(e[i]),e2[i])
        #    print(mean)
        return np.sqrt(mean)
    else:
        return np.nan


# In[ ]:


yt = np.array([1, 2, 3, 4, 1])
yp = np.array([1, 2, 1, 1, 1])

print(np.mean(yt))
print(RMSE(yt,yp))
print(RMSE(yt,yp, norm=True))
print(np.sqrt( ( ((1-1)/1)**2 + ((2-2)/2)**2 + ((3-1)/3)**2 + ((4-1)/4)**2 + ((1-1)/1)**2 )/5 ) )
print(RMSE(yt,yp, ALL= True, THETA = 3))


# In[ ]:


def MAE(ytrue, ypred, THETA=False, norm=False, ALL=False):
    if ALL == True:
        return {"MAE"    : MAE(ytrue, ypred),
                "MAPE"   : MAE(ytrue, ypred, norm=True),
                "MAEat"  : MAE(ytrue, ypred, THETA=THETA),
                "MAPEat" : MAE(ytrue, ypred, THETA=THETA, norm=True)
               }
    
    if THETA is False:
        includes = np.ones(len(ytrue), dtype=bool)
    else:
        includes = ytrue >= THETA
    
    if includes.any():
        yt = ytrue[includes]
        yp = ypred[includes]
        e = yt - yp
        if norm == True:
            e = e/yt
        eabs = np.abs(e)
        mean = np.mean(eabs)
        return mean
    else:
        return np.nan


# ## Quantile Metrics & Interval Coverage

# In[ ]:


#ytrue = actual
#ypred = quantile_values
#quantile_probs -> real quantiles
def quantile_metrics(actual, quantile_probs, quantile_values):
    actual = actual.reshape(-1)
    quantile_losses = []
    
    for idx in range(len(quantile_probs)):
        #print(actual.shape)
        #print(quantile_values.shape)
        #print(quantile_values[:,idx].shape)
        res = actual - quantile_values[:,idx]
        #print(res.shape)
        q = quantile_probs[idx]
        t=np.maximum(q*res, (q-1.0)*res)
        #print(t.shape)
        #print("######")
        
        qloss = np.mean(t) #pinball loss
        #if q == 0.9:
        #    print(actual[-15:])
        #    print(quantile_values[:,idx][-15:])
        #    print(res[-15:])
        #    print(t[-15:])
        quantile_losses.append(qloss)

    quantile_losses = np.array(quantile_losses)
    return np.mean(quantile_losses), quantile_losses


# In[ ]:


def IC_metrics(inf_limit, sup_limit, nominal_sig, true_values, base_prediction=0.0):
    inf_limit = inf_limit.flatten()
    sup_limit = sup_limit.flatten()
    true_values = true_values.flatten()
    base_prediction = base_prediction.flatten()
    
    EPS_CONST = np.finfo(float).eps
    rho = nominal_sig
    cov_prob = 0.0
    
    excess = []
    
    for idx in range(len(true_values)):
            one_excess = 0.0
            if true_values[idx] > sup_limit[idx]:
                    one_excess = (2.0/rho)*(true_values[idx] - sup_limit[idx])
            elif true_values[idx] < inf_limit[idx]:
                    one_excess = (2.0/rho)*(inf_limit[idx] - true_values[idx])
            else: 
                    cov_prob += 1.0

            excess.append(one_excess)

    excess = np.array(excess)
    MIS = sup_limit - inf_limit + excess #mean interval score
    MSIS =  np.divide(MIS, np.abs(true_values - base_prediction) + EPS_CONST) #mean scaled interval score
    #MSIS =  np.divide(MIS, np.abs(true_values) + EPS_CONST) #mean scaled interval score
    lenght = sup_limit - inf_limit

    #return np.mean(MIS), np.mean(MSIS), np.mean(cov_prob), np.mean(lenght)
    return np.mean(MIS), np.mean(MSIS), cov_prob/len(true_values), np.mean(lenght)


# In[ ]:


#def RMSEat(theta, ytrue, ypred):
#    includes = ytrue >= theta
#    if includes.any():
#        return math.sqrt(mean_squared_error(ytrue[includes], ypred[includes] ))
#    else:
#        return np.nan


# In[ ]:


#def MAEat(theta, ytrue, ypred):
#    includes = ytrue >= theta
#    if includes.any():
#        return mean_absolute_error(ytrue[includes], ypred[includes] )
#    else:
#        return np.nan


# # Lossses
# ## LossAtTHETA

# In[ ]:


def lossAtTHETA(scaled_THETA, ytrue, ypred):
    loss = 0
    e = ytrue - ypred
    loss = K.sum(K.square( e*K.maximum( K.sign(ytrue-scaled_THETA), 0) ), axis=-1)

    return loss


# In[ ]:


np.sign(61-(61-K.epsilon()))


# In[ ]:


# Loss Function
def custom_loss(ytrue, ypred):
    loss = 0
    #for i in range(ylen):
    e = ytrue - ypred
    loss += K.mean(K.square( e )*K.clip( ytrue-60, 0, 1 ), axis=-1)
    
    #loss = K.mean(K.square(ytrue[:, 0]-ypred[:, 0]), axis=-1)
        
    #for k in range(len(quantiles)):
    #    q = quantiles[k]
    #    e = (ytrue[:, ylen+k]-ypred[:, ylen+k])
    #    loss += K.mean(q*e + K.clip(-e, K.epsilon(), np.inf), axis=-1)
    return loss


# ## loss2YatTHETA

# In[ ]:


def loss2YatTHETA(scaled_THETA, ytrue, ypred):
    scaled_THETA -= K.epsilon()
    e1 = ytrue[:,0] - ypred[:,0]
    e2 = ytrue[:,1] - ypred[:,1]
    
    # <= THETA
    #loss1 = K.mean(K.square(ytrue[:,0]- ypred[:,0]))
    loss1 = K.sum(K.square( e1*(1-K.maximum( K.sign(ytrue[:,0]-scaled_THETA), 0)) ), axis=-1)
    loss1 = loss1/K.maximum(1., K.sum(1-K.maximum( K.sign(ytrue[:,0]-scaled_THETA), 0)))
    
    # > THETA
    loss2  = K.sum(K.square( e2*(K.maximum( K.sign(ytrue[:,1]-scaled_THETA), 0)) ), axis=-1)
    loss2 = loss2/K.maximum(1., K.sum(K.maximum( K.sign(ytrue[:,1]-scaled_THETA), 0)))
    
    #mean
    loss3 = 0
    #loss3 = K.mean(K.square(ytrue[:,2]- ypred[:,2]))
    
    return loss1 + loss2 + loss3


# In[ ]:


def loss2YatMEAN(scaled_THETA, ytrue, ypred):
    scaled_THETA -= K.epsilon()
    e1 = ytrue[:,0] - ypred[:,0]
    e2 = ytrue[:,1] - ypred[:,1]
    
    # <= THETA
    #loss1 = K.mean(K.square(ytrue[:,0]- ypred[:,0]))
    loss1 = K.sum(K.square( e1*(1-K.maximum( K.sign(ytrue[:,0]-scaled_THETA), 0)) ), axis=-1)
    loss1 = loss1/K.maximum(1., K.sum(1-K.maximum( K.sign(ytrue[:,0]-scaled_THETA), 0)))
    
    # > THETA
    loss2  = K.sum(K.square( e2*(K.maximum( K.sign(ytrue[:,1]-scaled_THETA), 0)) ), axis=-1)
    loss2 = loss2/K.maximum(1., K.sum(K.maximum( K.sign(ytrue[:,1]-scaled_THETA), 0)))
    
    #mean
    loss3 = 0
    #loss3 = K.mean(K.square(ytrue[:,2]- ypred[:,2]))
    
    return loss1 + loss2 + loss3


# # Clasificador

# ## ClassifierLSTM

# In[ ]:


def ClassifierLSTM(Config): #, AGREGADOS, TARGET, THETA, FUTURE, PAST, FEATURES, CUT, BAN, TIMESTEP, OVERLAP, BATCH_SIZE, TRAINPCT, OVERWRITE_MODEL, MODEL_NAME, LAYERS, EPOCHS, TIMEDIST):
    np.random.seed(123)
    

    TIMESTEP = Config['TIMESTEP']
    TIMEDIST = Config['TIMEDIST']
    Config['SHIFT'] = Config['FUTURE'] * -1
    
    scalers = {}
    ic = {"scalers": scalers}
    
    complete_dataset, ylabels, Yscaler, h24scaler = import_merge_and_scale(Config, verbose=False)
    ic["complete_dataset"] = complete_dataset
    ic["ylabels"] = ylabels
    scalers['Yscaler'] = Yscaler
    scalers["h24scaler"] = h24scaler
    
    y_len = len(ic["ylabels"])
    ic["y_len"] = y_len
    
    # TEST - Agregando datos del día siguiente - Son lineas COMENTABLES
    #dataset = pd.concat([dataset[dataset.columns[:-1]], dataset[["TEMP","UVB"]].shift(-1), dataset[dataset.columns[-1:]]], axis=1)
    #dataset.columns = dataset.columns[:-3].tolist() + ["TEMP2","UVB2","y"]
    #dataset[["TEMP","UVB"]] = dataset[["TEMP","UVB"]].shift(-1)
    
    data, features = select_features(ic, Config)
    ic["data"] = data
    ic["features"] = features
    
    ic["f_len"] = len(ic["features"])
    
    ic["secuencias"] = obtener_secuencias(ic)
    
    examples, y_examples, dateExamples = make_examples(ic, Config, verbose=False)
    ic["examples"] = examples
    ic["y_examples"] = y_examples
    ic["dateExamples"] = dateExamples
    
    trainX, trainY, testX, testY, dateTrain, dateTest = make_traintest(ic, Config, verbose=False)
    ic["trainX"] = trainX
    ic["trainY"] = trainY
    ic["testX"] = testX
    ic["testY"] = testY
    ic["dateTrain"] = dateTrain
    ic["dateTest"] = dateTest
    
    
    
    
    print("trainY.shape, ",ic["trainY"].shape)
    if TIMEDIST == True:
        ic["trainY"] = ic["trainY"].reshape(-1, TIMESTEP, y_len )
        ic["testY"] = ic["testY"].reshape(-1, TIMESTEP, y_len )
    else:
        ic["trainY"] = ic["trainY"][:,-1]
        ic["testY"] = ic["testY"][:,-1]
    print("trainY.shape, ",ic["trainY"].shape)
    
    EXAMPLES = Config["USE_EXAMPLES"]
    if EXAMPLES != None:
        print("jiji",ic["trainY"].shape)
        ic["trainY"] = EXAMPLES["trainY"].copy()
        ic["testY"] = EXAMPLES["testY"].copy()
        print("jiji",ic["trainY"].shape)
    
    
    cModel, file_name = ClassModel(ic, Config)
    ic["model"] = cModel
    ic["file_name"] = file_name
    
    trainPred, testPred = ClassPredict( ic, Config)
    ic["trainPred"] = trainPred
    ic["testPred"] = testPred
    
    print("classif,", np.sum(trainPred), np.sum(ic["trainY"]))
    print("classif,", np.sum(testPred), np.sum(ic["testY"]))
    
    #scalers = {"Yscaler":Yscaler,"h24scaler":h24scaler}
    #d = {"data":data, "trainX":trainX, "trainY":trainY, "testX":testX, "testY":testY, "dateTrain":dateTrain, "dateTest":dateTest,
            #"trainPred":ftrp, "trainYtrue":ftry, "testPred":ftep, "testYtrue":ftey,
    #        "modelName":MODEL_NAME, "scalers":scalers}
    return ic


# ## ClassPredict

# In[ ]:


def ClassPredict(internalConfig, Config):
    np.random.seed(123)
    set_random_seed(2)
    cModel = internalConfig["model"]
    trainX = internalConfig["trainX"]
    trainY = internalConfig["trainY"]
    testX = internalConfig["testX"]
    testY = internalConfig["testY"]
    
    BATCH_SIZE = Config["BATCH_SIZE"]
    TIMEDIST = Config["TIMEDIST"]
    PRED_TYPE = Config["PRED_TYPE"]
    
    # Predictions
    print("Classifier Prediction ...")
    trainPred = cModel.predict(trainX, batch_size = BATCH_SIZE)
    testPred = cModel.predict(testX, batch_size = BATCH_SIZE)
    
    # if TimeDistributed
    if TIMEDIST == True:
        pass
    else:
        print(trainPred.shape)
        print(trainY.shape)
        trainPred = trainPred[:, 0, None]
        trainY = trainY[:, 0, None]
        testPred = testPred[:, 0, None]
        testY = testY[:, 0, None]
        print(trainPred.shape)
        print(trainY.shape)
    
    
    print("classTrainPred", np.sum(trainPred), np.sum(trainY))
    print("classTestPred", np.sum(testPred), np.sum(testY))
    #for x,y in zip(trainPred, trainY):
    #    print((x>= np.mean(trainPred))*1, y,x )
    ####trainPred = np.round(trainPred)
    ####testPred = np.round(testPred)
    if PRED_TYPE == "hard":
        thetita = 0.5 #np.mean(trainPred)
        thetita = np.mean(trainPred)
        trainPred = (trainPred >= thetita)*1
        testPred = (testPred >= thetita)*1
    elif PRED_TYPE == "soft":
        # se mantiene igual
        pass
    print("classTrainPred", np.sum(trainPred), np.sum(trainY))
    print("classTestPred", np.sum(testPred), np.sum(testY))
    
    #print(trainY)
    #print(trainPred)
    # Invert Predictions
    #trainPred = Yscaler.inverse_transform(trainPred)
    #trainYinv = Yscaler.inverse_transform(trainY)
    #testPred = Yscaler.inverse_transform(testPred)
    #testYinv = Yscaler.inverse_transform(testY)
    
    # calculate root mean squared error
    if PRED_TYPE == 'hard':
        print("Calculando Accuracy")
        trainScore = accuracy_score(trainY, trainPred)
        print('    Train Accuracy: %.2f%%' % (trainScore*100))
        testScore = accuracy_score(testY, testPred)
        print('    Test Accuracy: %.2f%%' % (testScore*100))
    elif PRED_TYPE == 'soft':
        print("Calculando Accuracy")
        print("Not for PRED_TYPE='soft'")
    
    
    #join_index(trainPred,)
    return trainPred, testPred
    
    
    
    


# ## ClassModel

# In[ ]:


def ClassModel(internalConfig, Config ):
    np.random.seed(123)
    set_random_seed(2)
    
    trainX = internalConfig["trainX"]
    trainY = internalConfig["trainY"]
    #ylen = internalConfig["ylen"]
    features = internalConfig["features"]
    f_len = internalConfig["f_len"]
    #THETA = Config["THETA"]
    #FUTURE = Config["FUTURE"]
    #PAST = Config["PAST"]
    #TARGET = Config["TARGET"]
    #TRAINPCT = Config["TRAINPCT"]
    #MODEL_NAME = Config["MODEL_NAME"]
    #TIMEDIST = Config["TIMEDIST"]
    BATCH_SIZE = Config["BATCH_SIZE"]
    TIMESTEP = Config["TIMESTEP"]
    OVERWRITE_MODEL = Config["OVERWRITE_MODEL"]
    LAYERS = Config["LAYERS"]
    EPOCHS = Config["EPOCHS"]
    DROP_RATE = Config["DROP_RATE"]
    
    
    print("in ClassModel ...")
    print("    trainX.shape, ", trainX.shape)
    print("    trainY.shape", trainY.shape)
    
    #strFEATURES = str(features).replace("'","")
    #strTD = "TD"*TIMEDIST
    #FILE_NAME = "./%s/%s-%.2f-(%d,%d,%d)-%s-%.3f-%s-%s-%d-(%s,%s)-%s.h5"%(MODELS_FOLDER, MODEL_NAME, THETA, BATCH_SIZE, TIMESTEP, f_len, strFEATURES, TRAINPCT, TARGET, LAYERS, EPOCHS, str(PAST), FUTURE, strTD)
    #FILE_NAME = FILE_NAME.replace(" ","")
    FILE_NAME = create_filename(internalConfig, Config)
    
    
    try:
        if OVERWRITE_MODEL == False:
            print("Loading Model...")
            cModel = load_model(FILE_NAME)
            print(FILE_NAME + " loaded =)")
            print("LISTO")
        else:
            print("Reentrenando "+ FILE_NAME)
            load_model("noexisto_nijamas_existire.h5")
    except:
        print("batch_input_shape:(%d,%d,%d)"%(BATCH_SIZE,TIMESTEP,f_len))
        cModel = Sequential()
        for Neurons in LAYERS[:-1]:
            cModel.add(LSTM(Neurons, activation="relu", input_shape=(TIMESTEP, f_len), return_sequences=True))
            cModel.add(Dropout(rate=DROP_RATE))
        cModel.add(LSTM(LAYERS[-1], activation="relu", input_shape=(TIMESTEP, f_len), return_sequences=False))
        cModel.add(Dropout(rate=DROP_RATE))
        cModel.add(Dense( 1, activation='sigmoid'))
        cModel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        #cModel.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        cModel.fit( trainX, trainY, epochs=EPOCHS, batch_size=BATCH_SIZE  )
        
        
        cModel.save(FILE_NAME)
        print(FILE_NAME + " saved. = )")
    
    return cModel, FILE_NAME


# ## Pruebas

# In[ ]:


STATION = "Las_Condes"
ClassifierLSTMconfig={
                    # import_merge_and_scale()
                    "SCALER" : preprocessing.StandardScaler,
                 "AGREGADOS" : [],#["O3","TEMP","WS","RH"],    #["ALL"] #Horas de los maximos que se quieren agregar.
                   "PRECALC" : precalcular_agregados(STATION),
                    "TARGET" : "EC",
                     "THETA" : 61,
                    "FUTURE" : 1,
                      "PAST" : False,
                    
                    # select_features()
            "FIXED_FEATURES" : [], # empty list means that the CUT will be used
                       "CUT" : 0.26,
                       "BAN" : [],#["countEC"],#["countEC", "EC"],# []
                    
                    #make_examples()
                  "TIMESTEP" : 14,
                   "OVERLAP" : True,
                    
                    #make_traintest()
                   "SHUFFLE" : True,
                "BATCH_SIZE" : 16,
                  "TRAINPCT" : 0.85,
                    
                    #myLSTM()
           "OVERWRITE_MODEL" : False,
                "MODEL_NAME" : "ClassifierLSTMModel",
                    "LAYERS" : [4,4],
                    "EPOCHS" : 100,
                 "DROP_RATE" : 0.4,
                  "TIMEDIST" : False,  #SIN IMPLEMENTAR
                    }
#ClassifierOutput = ClassifierLSTM( ClassifierLSTMconfig )


# In[ ]:


#co = ClassifierOutput
#len(co["trainY"])


# # LSTM
# LSTM con función de pérdida de 'mean_squared_error'.

# ## myLSTM

# In[ ]:


def myLSTM(Config): #, AGREGADOS, TARGET, PRECALC, THETA, FUTURE, PAST, FEATURES, CUT, BAN, TIMESTEP, OVERLAP, BATCH_SIZE, TRAINPCT, OVERWRITE_MODEL, MODEL_NAME, LAYERS, EPOCHS, TIMEDIST, Yx):
    #np.random.seed(123)
    #set_random_seed(2)
    
    FOLDS_TVT = Config["FOLDS_TVT"]
    TIMESTEP = Config['TIMESTEP']
    #TIMEDIST = Config['TIMEDIST']
    Config['SHIFT'] = Config['FUTURE'] * -1
    
    #SHIFT = FUTURE*-1
    #PRECALC = precalcular_agregados()
    
    
    scalers = {}
    ic = {"scalers": scalers}
    
    complete_dataset, ylabels, Yscaler, h24scaler = import_merge_and_scale(Config, verbose=False)
    ic["complete_dataset"] = complete_dataset
    ic["ylabels"] = ylabels
    scalers['Yscaler'] = Yscaler
    scalers["h24scaler"] = h24scaler
    
    # TEST - Agregando datos del día siguiente - Son lineas COMENTABLES
    #dataset = pd.concat([dataset[dataset.columns[:-1]], dataset[["TEMP","UVB"]].shift(-1), dataset[dataset.columns[-1:]]], axis=1)
    #dataset.columns = dataset.columns[:-3].tolist() + ["TEMP2","UVB2","y"]
    #dataset[["TEMP","UVB"]] = dataset[["TEMP","UVB"]].shift(-1)
    
    y_len = len(ic["ylabels"])
    ic["y_len"] = y_len
    
    
    data, features = select_features(ic, Config)
    ic["data"] = data
    ic["features"] = features
    
    ic["f_len"] = len(ic["features"])
    
    ic["secuencias"] = obtener_secuencias(ic)
    
    #data, FEATURES = select_features(dataset, ylabels, FEATURES, CUT, BAN)
    #F_LEN = len(FEATURES)
    
    #strFEATURES = str(FEATURES).replace("'","")
    #strTD = "TD"*TIMEDIST
    #FILE_NAME = "./%s/%s-(%d,%d,%d)-%s-%.3f-%s-%s-%d-(%s,%s)-%s.h5"%(MODELS_FOLDER,MODEL_NAME, BATCH_SIZE, TIMESTEP, F_LEN, strFEATURES, TRAINPCT, TARGET, LAYERS, EPOCHS, str(PAST), FUTURE, strTD)
    #FILE_NAME = FILE_NAME.replace(" ","")
    
    
    #SECUENCIAS = obtener_secuencias(data)
    #examples, y, date_index = make_examples(data, SECUENCIAS, ylabels, TIMESTEP, OVERLAP, verbose=False)
    #trainX, trainY, testX, testY, dateTrain, dateTest = make_traintest(examples, y, date_index, BATCH_SIZE, TRAINPCT, verbose=False)
    
    examples, y_examples, dateExamples = make_examples(ic, Config, verbose=False)
    ic["examples"] = examples
    ic["y_examples"] = y_examples
    ic["dateExamples"] = dateExamples
    
    if FOLDS_TVT == False:
        tvtDict = make_traintest(ic, Config, verbose=False)
        for key in tvtDict:
            ic[key] = tvtDict[key]
        #trainX, trainY, validX, validY, testX, testY, dateTrain, dateValid, dateTest = make_traintest(ic, Config, verbose=False)
        #ic["trainX"] = trainX
        #ic["trainY"] = trainY
        #ic["validX"] = validX
        #ic["validY"] = validY
        #ic["testX"] = testX
        #ic["testY"] = testY
        #ic["dateTrain"] = dateTrain
        #ic["dateValid"] = dateValid
        #ic["dateTest"] = dateTest
    else:
        list_tvtDict = make_folds_TVT(ic, Config)
        for key in list_tvtDict:
            ic[key] = list_tvtDict[key]
    
    list_models , file_name = myLSTMModel(ic, Config)
    
    if FOLDS_TVT == False:
        ic["model"] = list_models[0]
        ic["list_models"] = [ ic["model"] ]
    else:
        ic["list_models"] = list_models
    
    ic["file_name"] = file_name #solo es el ultimo modelo
    
    
    
    
    detail = myLSTMPredict(ic, Config)
    ic["detail"] = detail
    #trainPred, validPred, testPred = myLSTMPredict(ic, Config)
    #ic["trainPred"] = trainPred
    #ic["validPred"] = validPred
    #ic["testPred"] = testPred
    
    return ic


# ## myLSTMPredict

# In[ ]:


def myLSTMPredict(internalConfig, Config):
    TIMEDIST = Config["TIMEDIST"]
    BATCH_SIZE = Config["BATCH_SIZE"]
    THETA = Config["THETA"]
    TARGET = Config["TARGET"]
    FOLDS_TVT = Config["FOLDS_TVT"]
    GRAPH = Config["GRAPH"]
    moreTHETA = Config["moreTHETA"]
    Yx = Config["Yx"]
    Yscaler = internalConfig["scalers"]["Yscaler"]
    list_models = internalConfig["list_models"]
    
    if FOLDS_TVT == False:
        list_trainX = [ internalConfig[   "trainX"] ]
        list_trainY = [ internalConfig[   "trainY"] ]
        list_validX = [ internalConfig[   "validX"] ]
        list_validY = [ internalConfig[   "validY"] ]
        list_testX  = [ internalConfig[   "testX" ] ]
        list_testY  = [ internalConfig[   "testY" ] ]
        list_dateTrain = [ internalConfig["dateTrain"] ]
        list_dateValid = [ internalConfig["dateValid"] ]
        list_dateTest  = [ internalConfig[ "dateTest"] ]
        
    else:
        list_trainX = internalConfig[   "list_trainX"]
        list_trainY = internalConfig[   "list_trainY"]
        list_validX = internalConfig[   "list_validX"]
        list_validY = internalConfig[   "list_validY"]
        list_testX = internalConfig[    "list_testX"]
        list_testY = internalConfig[    "list_testY"]
        list_dateTrain = internalConfig["list_dateTrain"]
        list_dateValid = internalConfig["list_dateValid"]
        list_dateTest = internalConfig[ "list_dateTest"]
    
    
    # RMSE
    trainRMSE = []
    validRMSE = []
    testRMSE  = []
    
    # MAE
    trainMAE = []
    validMAE = []
    testMAE  = []
    
    scores_detail = {
                "train": {'quantity':[]},
                "valid": {'quantity':[]},
                "test" : {'quantity':[]},
                }
    
    list_df = []
    
    print("Calculando Predicciones")

    for fold in range(0,len(list_trainX)):
        trainX = list_trainX[fold]
        trainY = list_trainY[fold]
        validX = list_validX[fold]
        validY = list_validY[fold]
        testX  =  list_testX[fold]
        testY  =  list_testY[fold]
        dateTrain = list_dateTrain[fold]
        dateValid = list_dateValid[fold]
        dateTest  = list_dateTest[fold]
        
        
        
        
        
        classicModel = list_models[fold]
    
    
        # Predictions
        
        trainPred = classicModel.predict(trainX)
        classicModel.reset_states()
        validPred = classicModel.predict(validX)
        classicModel.reset_states()
        testPred = classicModel.predict(testX)
        classicModel.reset_states()
        
        #TimeDistributed
        T = Yx
        if TIMEDIST == True:
            #print(trainPred.shape)
            #print(trainY.shape)
            trainPred = trainPred[:,-1, T].reshape(-1, 1)
            trainY = trainY[:,-1,0].reshape(-1, 1)
            validPred = validPred[:,-1, T].reshape(-1, 1)
            validY = validY[:,-1,0].reshape(-1, 1)
            testPred = testPred[:,-1, T].reshape(-1, 1)
            testY = testY[:,-1,0].reshape(-1, 1)
            #print(trainPred.shape)
            #print(trainPred.shape)
        else:
            #print(trainPred.shape)
            #print(trainY.shape)
            trainPred = trainPred[:, T, None]
            trainY = trainY[:, T, None]
            validPred = validPred[:, T, None]
            validY = validY[:, T, None]
            testPred = testPred[:, T, None]
            testY = testY[:, T, None]
            #print(trainPred.shape)
            #print(trainY.shape)
        
        # Invert Predictions
        trainPred = Yscaler.inverse_transform(trainPred)
        trainYinv = Yscaler.inverse_transform(trainY)
        
        validPred = Yscaler.inverse_transform(validPred)
        validYinv = Yscaler.inverse_transform(validY)
        
        testPred = Yscaler.inverse_transform(testPred)
        testYinv = Yscaler.inverse_transform(testY)
        
        # stats
        num_train = len(trainX)
        num_valid = len(validX)
        num_test = len(testX)
        
        num_THETA_train = np.sum(trainYinv >= THETA)
        num_THETA_valid = np.sum(validYinv >= THETA)
        num_THETA_test = np.sum(testYinv >= THETA)
        
        scores_detail["train"]['quantity'].append( (num_train, num_THETA_train) )
        scores_detail["valid"]['quantity'].append( (num_valid, num_THETA_valid) )
        scores_detail["test"]['quantity'].append(  (num_test, num_THETA_test)   )
        
        
        # calculate root mean squared error
        yt, yp = trainYinv, trainPred
        scores = RMSE(yt, yp, THETA=THETA, ALL=True)
        scores = [scores["RMSE"], scores["RMSPE"], scores["RMSEat"], scores["RMSPEat"] ]
        for t in moreTHETA:
            scores.append( RMSE(yt, yp, THETA=t) )
            scores.append( RMSE(yt, yp, THETA=t, norm=True) )
        trainRMSE.append( scores )
        yt, yp = validYinv, validPred
        scores = RMSE(yt, yp, THETA=THETA, ALL=True)
        scores = [scores["RMSE"], scores["RMSPE"], scores["RMSEat"], scores["RMSPEat"] ]
        for t in moreTHETA:
            scores.append( RMSE(yt, yp, THETA=t) )
            scores.append( RMSE(yt, yp, THETA=t, norm=True) )
        validRMSE.append( scores )
        yt, yp = testYinv, testPred
        scores = RMSE(yt, yp, THETA=THETA, ALL=True)
        scores = [scores["RMSE"], scores["RMSPE"], scores["RMSEat"], scores["RMSPEat"] ]
        for t in moreTHETA:
            scores.append( RMSE(yt, yp, THETA=t) )
            scores.append( RMSE(yt, yp, THETA=t, norm=True) )
        testRMSE.append( scores )
        a,b = trainRMSE[-1][0:2]
        c,d = validRMSE[-1][0:2]
        e,f = testRMSE[-1][0:2]
        print("fold %s score (RMSE,RMSPE): (%.2f, %.2f) (%.2f, %.2f) (%.2f, %.2f)"%(fold, a,b,c,d,e,f))
        
        #scores = RMSE(validYinv, validPred, THETA=THETA, ALL=True)
        #validRMSE.append( [scores["RMSE"], scores["RMSPE"], scores["RMSEat"], scores["RMSPEat"]] )
        #scores = RMSE(testYinv, testPred, THETA=THETA, ALL=True)
        #testRMSE.append(  [scores["RMSE"], scores["RMSPE"], scores["RMSEat"], scores["RMSPEat"]] )
        
        
        #trainScore = RMSE(trainYinv, trainPred, 61)
        #trainRMSE[-1].append( trainScore )
        #validScore = RMSE(validYinv, validPred, 61)
        #validRMSE[-1].append( validScore )
        #testScore = RMSE(testYinv, testPred, 61)
        #testRMSE[-1].append( testScore )
        
        
        #MAE
        yt, yp = trainYinv, trainPred
        scores = MAE(yt, yp, THETA=THETA, ALL=True)
        scores = [scores["MAE"], scores["MAPE"], scores["MAEat"], scores["MAPEat"] ]
        for t in moreTHETA:
            scores.append( MAE(yt, yp, THETA=t) )
            scores.append( MAE(yt, yp, THETA=t, norm=True) )
        trainMAE.append( scores )
        #scores = MAE(trainYinv, trainPred, THETA=THETA, ALL=True)
        #trainMAE.append( [scores["MAE"], scores["MAPE"], scores["MAEat"], scores["MAPEat"]] )
        yt, yp = validYinv, validPred
        scores = MAE(yt, yp, THETA=THETA, ALL=True)
        scores = [scores["MAE"], scores["MAPE"], scores["MAEat"], scores["MAPEat"] ]
        for t in moreTHETA:
            scores.append( MAE(yt, yp, THETA=t) )
            scores.append( MAE(yt, yp, THETA=t, norm=True) )
        validMAE.append( scores )
        #scores = MAE(validYinv, validPred, THETA=THETA, ALL=True)
        #validMAE.append( [scores["MAE"], scores["MAPE"], scores["MAEat"], scores["MAPEat"]] )
        yt, yp = testYinv, testPred
        scores = MAE(yt, yp, THETA=THETA, ALL=True)
        scores = [scores["MAE"], scores["MAPE"], scores["MAEat"], scores["MAPEat"] ]
        for t in moreTHETA:
            scores.append( MAE(yt, yp, THETA=t) )
            scores.append( MAE(yt, yp, THETA=t, norm=True) )
        testMAE.append( scores )
        #scores = MAE(testYinv, testPred, THETA=THETA, ALL=True)
        #testMAE.append( [scores["MAE"], scores["MAPE"], scores["MAEat"], scores["MAPEat"]] )
        

        if GRAPH == True:
            print("Graficando...")
            df1 = join_index(dateTrain,trainYinv, "Y")
            df2 = join_index(dateTrain,trainPred, "f train")
            
            df3 = join_index(dateValid,validYinv, "Y")
            df4 = join_index(dateValid,validPred, "f validation")
            
            df5 = join_index(dateTest,testYinv, "Y")
            df6 = join_index(dateTest,testPred, "f test")
            
            ydf = pd.concat([df1, df3, df5], axis=0)
            df = pd.concat( [ydf, df4, df2, df6] ,axis=1)
            
            #traindf = pd.concat([df1,df2], axis=1)
            #testdf = pd.concat([df3,df4], axis=1)
            
            #df = pd.concat([traindf,testdf])
            
            #fecha_test = testdf.index[0]
            #df.asfreq("D").iplot(title = "%s - %s"%(TARGET, LAYERS), vline=[fecha_test], hline=[THETA])
            df.asfreq("D").iplot(title = "%s"%(TARGET), hline=[THETA, df["Y"].mean()], vline=[dateValid[0],dateTest[0]])
            
            df = df.asfreq("D")
            
            list_df.append(df)
        
    print("### LSTM ###")
    
    labels = ["RMSE", "RMSPE", "RMSEat%s"%THETA, "RMSPEat%s"%THETA]
    for t in moreTHETA:
        labels.append("RMSEat%s"%t)
        labels.append("RMSPEat%s"%t)
    maxlen = max(map(len,labels))
    trainMeans = np.nanmean(trainRMSE, axis=0)
    validMeans = np.nanmean(validRMSE, axis=0)
    testMeans  = np.nanmean(testRMSE, axis=0)
    print("{:>{maxlen}}{:>8}{:>8}{:>8}".format("","Train","Valid","Test",maxlen=maxlen))
    for i in range(len(labels)):
        print( ("{0:>{maxlen}}{1:8.2f}{2:8.2f}{3:8.2f}".format (labels[i], trainMeans[i], validMeans[i], testMeans[i], maxlen=maxlen)) )
        
        scores_detail["train"][labels[i]] = np.array(trainRMSE)[:,i]
        scores_detail["valid"][labels[i]] = np.array(validRMSE)[:,i]
        scores_detail["test"][labels[i]]  = np.array(testRMSE)[:,i]
        
    print("\n")
    
    labels = ["MAE", "MAPE", "MAEat%s"%THETA, "MAPEat%s"%THETA]
    for t in moreTHETA:
        labels.append("RMSEat%s"%t)
        labels.append("RMSPEat%s"%t)
    maxlen = max(map(len,labels))
    trainMeans = np.nanmean(trainMAE, axis=0)
    validMeans = np.nanmean(validMAE, axis=0)
    testMeans  = np.nanmean( testMAE, axis=0)
    print("{:>{maxlen}}{:>8}{:>8}{:>8}".format("","Train","Valid","Test",maxlen=maxlen))
    for i in range(len(labels)):
        print( ("{0:>{maxlen}}{1:8.2f}{2:8.2f}{3:8.2f}".format (labels[i], trainMeans[i], validMeans[i], testMeans[i], maxlen=maxlen)) )
    
        scores_detail["train"][labels[i]] = np.array(trainMAE)[:,i]
        scores_detail["valid"][labels[i]] = np.array(validMAE)[:,i]
        scores_detail["test"][labels[i]]  = np.array(testMAE)[:,i]
    print("\n")
    
    
    if FOLDS_TVT == False:
        # PLOT
        internalConfig["trainYinv"] = trainYinv
        internalConfig["validYinv"] = validYinv
        internalConfig["testYinv"] = testYinv
    else:
        trainPred = "DUMMY"
        validPred = "DUMMY"
        testPred = "DUMMY"
    
    internalConfig["list_df"] = list_df
    
    
    return scores_detail


# In[ ]:


a = np.array([ [1,2,3],
               [4,5,6],
               [7,8,9]])
a[:,1]


# ## myLSTMModel

# In[ ]:


def myLSTMModel(internalConfig, Config):
    SEED = Config['SEED']
    np.random.seed(SEED)
    set_random_seed(SEED*SEED)
    
    ic = internalConfig
    
    TIMESTEP = Config['TIMESTEP']
    TIMEDIST = Config["TIMEDIST"]
    FOLDS_TVT = Config["FOLDS_TVT"]
    PATIENCE = Config["PATIENCE"]
    OWN_SAVE = Config["OWN_SAVE"]
    OWN_LOAD = Config["OWN_LOAD"]
    

    
    
    OVERWRITE_MODEL = Config["OVERWRITE_MODEL"]
    TARGET = Config["TARGET"]
    BATCH_SIZE = Config["BATCH_SIZE"]
    THETA = Config["THETA"]
    LAYERS = Config["LAYERS"]
    DROP_RATE = Config["DROP_RATE"]

    EPOCHS = Config["EPOCHS"]
    LOSS = Config["LOSS"]
    f_len = ic["f_len"]
    y_len = ic["y_len"]
    
    
    if FOLDS_TVT == False:
        #if TIMEDIST == True:
        #    ic["trainY"] = ic["trainY"].reshape(-1, TIMESTEP, y_len )
        #    ic["validY"] = ic["validY"].reshape(-1, TIMESTEP, y_len )
        #    ic["testY"] = ic["testY"].reshape(-1, TIMESTEP, y_len )
        #else:
        #    ic["trainY"] = ic["trainY"][:,-1]
        #    ic["validY"] = ic["validY"][:,-1]
        #    ic["testY"] = ic["testY"][:,-1]
        
        list_trainX = [ ic["trainX"] ]
        list_trainY = [ ic["trainY"] ]
        list_validX = [ ic["validX"] ]
        list_validY = [ ic["validY"] ]
    else:
        list_trainX = internalConfig['list_trainX']
        list_trainY = internalConfig['list_trainY']
        list_validX = internalConfig['list_validX']
        list_validY = internalConfig['list_validY']
    
    
    losses = []
    list_models = []
    
    LAYERS.reverse()
    DROP_RATE.reverse()
    
    for fold in range(0,len(list_trainX)):
        internalConfig['fold'] = fold + 1
        print("Using Fold index: %d/%d"%(fold,len(list_trainX)-1) )
        trainX = list_trainX[fold]
        trainY = list_trainY[fold]
        validX = list_validX[fold]
        validY = list_validY[fold]
        
    
        f_len = trainX.shape[-1]
        print("trainX.shape,",trainX.shape)
        print("trainY.shape,",trainY.shape)
        print("validX.shape,",validX.shape)
        print("validY.shape,",validY.shape)
        
    
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=PATIENCE, restore_best_weights=True)
        
        FILE_NAME = create_filename(ic, Config)
        
        try:
            if OVERWRITE_MODEL == False:
                print("Loading Model...")
                if OWN_LOAD:
                    if FOLDS_TVT == True:
                        last_FILE_NAME = "%s/%s-%s.h5"%(MODELS_FOLDER,OWN_LOAD, fold)
                        classicModel = load_model( last_FILE_NAME )
                    else:
                        last_FILE_NAME = "%s/%s.h5"%(MODELS_FOLDER,OWN_LOAD)
                        classicModel = load_model( last_FILE_NAME )
                else:
                    classicModel = load_model(FILE_NAME)
                    last_FILE_NAME = FILE_NAME
                
                list_models.append(classicModel)
                print(FILE_NAME + " loaded =)")
                print("LISTO")
            else:
                print("Reentrenando "+ FILE_NAME)
                load_model("noexisto_nijamas_existire.h5")
        except:
            print("batch_input_shape:(%d,%d,%d)"%(BATCH_SIZE,TIMESTEP,f_len))
            print("Loading Failed. Training again...")
            classicModel = Sequential()
            
            
            ## SIN stateful
            if len(LAYERS) == 2:
                classicModel.add(LSTM(LAYERS[1], activation="sigmoid", input_shape=(TIMESTEP, f_len), return_sequences=True))
                classicModel.add(Dropout(rate=DROP_RATE[1]))
            
            if TIMEDIST == True:
                classicModel.add(LSTM(LAYERS[0], activation="sigmoid", input_shape=(TIMESTEP, f_len), return_sequences=True))
                classicModel.add(Dropout(rate=DROP_RATE[0]))
                classicModel.add( TimeDistributed( Dense( y_len, activation="linear") ) )
            else:
                classicModel.add(LSTM(LAYERS[0], activation="sigmoid", input_shape=(TIMESTEP, f_len), return_sequences=False))
                classicModel.add(Dropout(rate=DROP_RATE[0]))
                classicModel.add(Dense(y_len, activation="linear"))
            
            if LOSS == "":
                classicModel.compile(loss='mean_squared_error', optimizer='adam')
            elif LOSS == "atTHETA":
                Yscaler = ic["scalers"]["Yscaler"]
                scaled_theta = Yscaler.transform([[ Config["THETA"] ]])
                classicModel.compile(loss=lambda y,f: lossAtTHETA(scaled_theta,y,f), optimizer='adam')
            
            classicModel.fit(trainX, trainY, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(validX,validY), callbacks=[es], verbose=1)#, shuffle=False)
            
            #classicModel.fit(trainX, trainY, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2)
            classicModel.save(FILE_NAME)
            if FOLDS_TVT == True:
                classicModel.save("%s/%s-%s.h5"%(MODELS_FOLDER, OWN_SAVE, fold))
            else:
                classicModel.save("%s/%s-%s.h5"%(MODELS_FOLDER, OWN_SAVE))
            print(FILE_NAME + " saved. = )")
            
            list_models.append(classicModel)
        
    return list_models, FILE_NAME
    #ic["model"] = classicModel
    #ic["file_name"] = FILE_NAME


# ## Pruebas
# Para cada configuración, si ya hay un modelo entrenado, éste será cargado, si no existe un modelo entrenado para dicha configuración, entonces se entrenará y guardará.

# In[ ]:


STATION, FILTER_YEARS, THETA = get_station("Parque_OHiggins")
LSTMconfig = {
                    # import_merge_and_scale()
                   "STATION" : STATION,
                    "SCALER" : preprocessing.StandardScaler,
                "IMPUTATION" : None,
                 "AGREGADOS" : [],#["O3","TEMP","WS","RH"],    #["ALL"] #Horas de los maximos que se quieren agregar.
                    "TARGET" : "O3",
                   "PRECALC" : precalcular_agregados(STATION),
                     "THETA" : THETA,
                 "moreTHETA" : [61],
                    "FUTURE" : 1,
                      "PAST" : False,
                    
                    # select_features()
              "FILTER_YEARS" : FILTER_YEARS,#[2004,2013], #En Veranos
                       "CUT" : 0.41,
            "FIXED_FEATURES" : ['CO', 'PM10', 'PM25', 'NO', 'NOX', 'WD', 'RH', 'TEMP', 'WS', 'UVA', 'UVB', 'O3'], # empty list means that the CUT will be used
                       "BAN" : ["countEC", "EC","O3btTHETA"], # []
                    
                    #make_examples()
                 "FOLDS_TVT" : True,
                  "TIMESTEP" : 14,
                   "OVERLAP" : True,
                    
                    #make_traintest() Ignored when FOLDS_TVT == True
                   "SHUFFLE" : False,
                  "TRAINPCT" : 0.85,
                   
                    
                    #myLSTM()
           "OVERWRITE_MODEL" : True,
                "MODEL_NAME" : "classicModel",
                    "LAYERS" : [19],
                  "DROP_RATE": [0.614707658384578],
                "BATCH_SIZE" : 16,
                    "EPOCHS" : 400,
                  "PATIENCE" : 20,
                  "TIMEDIST" : False,
                      "LOSS" : "", # "" or "atTHETA"
                     "GRAPH" : True,
                  "OWN_SAVE" : "MiModelo",
                  "OWN_LOAD" : None, #"MiModelo",
                    #Y to calc Error. For Test Only.
                        "Yx" : 0    # DEFAULT 0
                }
#myLSTMoutput = myLSTM(LSTMconfig)


# In[ ]:


#LSTMconfig["GRAPH"] = False
#LSTMconfig["moreTHETA"] = []#[89, 76, 61]
#a =myLSTMPredict(myLSTMoutput, LSTMconfig)


# ## Pruebas con varias seeds

# In[ ]:





# In[ ]:


STATION, FILTER_YEARS, THETA = get_station("POH_full")
LSTMconfig = {
                    # import_merge_and_scale()
                   "STATION" : STATION,
                    "SCALER" : preprocessing.StandardScaler,
                "IMPUTATION" : None,
                 "AGREGADOS" : [],#["O3","TEMP","WS","RH"],    #["ALL"] #Horas de los maximos que se quieren agregar.
                    "TARGET" : "O3",
                   "PRECALC" : precalcular_agregados(STATION),
                     "THETA" : THETA,
                 "moreTHETA" : [61],
                    "FUTURE" : 1,
                      "PAST" : False,
                    
                    # select_features()
              "FILTER_YEARS" : FILTER_YEARS,#[2004,2013], #En Veranos
                       "CUT" : 0.41,
            "FIXED_FEATURES" : ['CO', 'PM10', 'PM25', 'NO', 'NOX', 'WD', 'RH', 'TEMP', 'WS', 'UVA', 'UVB', 'O3'], # empty list means that the CUT will be used
                       "BAN" : ["countEC", "EC","O3btTHETA"], # []
                    
                    #make_examples()
                 "FOLDS_TVT" : True,
                  "TIMESTEP" : 21,
                   "OVERLAP" : True,
                    
                    #make_traintest() Ignored when FOLDS_TVT == True
                   "SHUFFLE" : False,
                  "TRAINPCT" : 0.85,
                   
                    
                    #myLSTM()
           "OVERWRITE_MODEL" : False,
                "MODEL_NAME" : "classicModel",
                    "LAYERS" : [32],
                  "DROP_RATE": [0.823807768638893],
                "BATCH_SIZE" : 16,
                    "EPOCHS" : 400,
                  "PATIENCE" : 20,
                  "TIMEDIST" : False,
                      "LOSS" : "", # "" or "atTHETA"
                     "GRAPH" : False,
                  "OWN_SAVE" : "MiModelo",
                  "OWN_LOAD" : None, #"MiModelo",
                    #Y to calc Error. For Test Only.
                        "Yx" : 0    # DEFAULT 0
                }

seeds = [123, 57, 872, 340, 77, 583, 101, 178, 938, 555]
all_scores = []
all_outputs = []
for s in seeds:
    LSTMconfig["SEED"] = s
    myLSTMoutput = myLSTM(LSTMconfig)
    all_outputs.append(myLSTMoutput)
    all_scores.append( myLSTMPredict(myLSTMoutput, LSTMconfig) )


# In[ ]:



all_metrics = ["RMSE", "RMSEat%s"%THETA]
for m in all_metrics:
    for d in ['train', 'valid', 'test']:
        fmeans = []
        for i in range( len(seeds) ):
            fmeans.append( np.nanmean(all_scores[i][d][m]) ) #mean over folds
        mean = np.mean(fmeans)
        std  = np.std(fmeans)
        print(m, d, mean, std)


# In[ ]:


import pickle
output = open("POH_all_scores.pkl", 'wb')
pickle.dump(all_scores, output)
output.close()


STOP


# In[ ]:





# # LSTM 2Y-Loss

# ## repeatY

# In[ ]:


def repeatY(internalConfig, cant):
    trainY = internalConfig["trainY"]
    testY = internalConfig["testY"]
    TIMEDIST = False
    
    print("Repeating Y ..")
    
    if TIMEDIST == True:
        pass
    else:
        trainYr = trainY[:, -1, :]
        testYr = testY[:, -1, :]
        
        trainY = trainY[:, -1, 0, None]
        testY = testY[:, -1, 0, None]
        
        #if cant > 1:
        for i in range(1,cant):
            #trainYq = np.hstack([trainYq, trainY[:,None]])
            #testYq = np.hstack([testYq, testY[:,None]])
            trainYr = np.hstack([trainYr, trainY[:]])
            testYr = np.hstack([testYr, testY[:]])
    
    print("    trainYq.shape, ", trainYr.shape)
    print("    testYq.shape, ", testYr.shape)
    
    return trainYr, testYr


# ## LSTM_2Y

# In[ ]:


def LSTM_2Y(Config): #, AGREGADOS, TARGET, PRECALC, THETA, FUTURE, PAST, FEATURES, CUT, BAN, TIMESTEP, OVERLAP, BATCH_SIZE, TRAINPCT, OVERWRITE_MODEL, MODEL_NAME, LAYERS, EPOCHS, TIMEDIST, Yx):
    scalers = {}
    ic = {"scalers": scalers}
    
    Config["CLASSIFIER_CONFIG"]["PRED_TYPE"] = Config["PRED_TYPE"]
    classifierOutput = ClassifierLSTM(Config["CLASSIFIER_CONFIG"])
    ic["classifierOutput"] = classifierOutput
    
    np.random.seed(123)
    set_random_seed(2)
    
    Config['SHIFT'] = Config['FUTURE'] * -1
    
    #SHIFT = FUTURE*-1
    #PRECALC = precalcular_agregados()
    
    
    
    
    complete_dataset, ylabels, Yscaler, h24scaler = import_merge_and_scale(Config, verbose=False)
    ic["complete_dataset"] = complete_dataset
    ic["ylabels"] = ylabels
    scalers['Yscaler'] = Yscaler
    scalers["h24scaler"] = h24scaler
    
    
    y_len = len(ic["ylabels"])
    ic["y_len"] = y_len
    
    
    data, features = select_features(ic, Config)
    ic["data"] = data
    ic["features"] = features
    
    ic["f_len"] = len(ic["features"])
    
    ic["secuencias"] = obtener_secuencias(ic)
    
    
   
    examples, y_examples, dateExamples = make_examples(ic, Config, verbose=False)
    ic["examples"] = examples
    ic["y_examples"] = y_examples
    ic["dateExamples"] = dateExamples
    
    # Merge with classifier
    
    
    
    
    trainX, trainY, testX, testY, dateTrain, dateTest = make_traintest(ic, Config, verbose=False)
    ic["trainX"] = trainX
    ic["trainY"] = trainY
    ic["testX"] = testX
    ic["testY"] = testY
    ic["dateTrain"] = dateTrain
    ic["dateTest"] = dateTest
    
    print("YY",trainY.shape)
    trainY, testY = repeatY(ic, 2)
    ic["trainY"] = trainY
    ic["testY"] = testY
    print("YY",trainY.shape)
    print(trainY)
    
    
    ### Model
    classicModel , file_name = LSTM_2YModel(ic, Config)
    ic["model"] = classicModel
    ic["file_name"] = file_name
    
    ### Prediction
    LSTM_2YPredict(ic, Config)
    #trainPred, testPred, df = LSTM_2YPredict(ic, Config)
    #ic["trainPred"] = trainPred
    #ic["testPred"] = testPred
    #ic["df"] = df
    
    return ic


# ## LSTM_2YPredict

# In[ ]:


def LSTM_2YPredict(internalConfig, Config):
    TIMEDIST = Config["TIMEDIST"]
    BATCH_SIZE = Config["BATCH_SIZE"]
    THETA = Config["THETA"]
    TARGET = Config["TARGET"]
    LAYERS = Config["LAYERS"]
    PRED_TYPE = Config["PRED_TYPE"]
    Yx = Config["Yx"]
    Yscaler = internalConfig["scalers"]["Yscaler"]
    trainX = internalConfig["trainX"]
    trainY = internalConfig["trainY"]
    testX = internalConfig["testX"]
    testY = internalConfig["testY"]
    dateTrain = internalConfig["dateTrain"]
    dateTest = internalConfig["dateTest"]
    classicModel = internalConfig["model"]
    classifierOutput = internalConfig["classifierOutput"]
    
    # Predictions
    print("Calculando Predicciones")
    trainPred = classicModel.predict(trainX, batch_size=BATCH_SIZE)
    classicModel.reset_states()
    testPred = classicModel.predict(testX, batch_size=BATCH_SIZE)
    classicModel.reset_states()
    
    #TimeDistributed
    ##T = Yx
    ##if TIMEDIST == True:
    ##    print(trainPred.shape)
    ##    print(trainY.shape)
    ##    trainPred = trainPred[:,-1, T].reshape(-1, 1)
    ##    trainY = trainY[:,-1,0].reshape(-1, 1)
    ##    testPred = testPred[:,-1, T].reshape(-1, 1)
    ##    testY = testY[:,-1,0].reshape(-1, 1)
    ##    print(trainPred.shape)
    ##    print(trainPred.shape)
    ##else:
    ##    print(trainPred.shape)
    ##    print(trainY.shape)
    ##    trainPred = trainPred[:, T, None]
    ##    trainY = trainY[:, T, None]
    ##    testPred = testPred[:, T, None]
    ##    testY = testY[:, T, None]
    ##    print(trainPred.shape)
    ##    print(trainY.shape)
    
    ###print("###", trainPred)
    ####print("###", trainY)
    ###
    classTrainPred = classifierOutput["trainPred"]
    classTrainDate = classifierOutput["dateTrain"]
    ###df1 = join_index(classTrainDate, classTrainPred, "classTrain")
    ###
    classTestPred = classifierOutput["testPred"]
    classTestDate = classifierOutput["dateTest"]
    ###df2 = join_index(classTestDate, classTestPred, "classTest")
    ###
    ###dfDate = pd.concat([df1,df2], axis=0)
    ###
    ###print( dateTrain.shape, trainPred.shape )    
    ###dfTrain = join_index(dateTrain, trainPred, "trainPred" )
    ###print(len(dfDate))
    ###print(dfTrain)
    ###dfTrain["classTest"] = dfDate
    ###print(dfTrain)
    
    #print("QQ",len(trainPred),len(classTrainPred))
    #print("QQ",len(testPred),len(classTestPred))
    
    if PRED_TYPE == "hard":
        ntrainPred = []
        for i in range(len(trainPred)):
            #ntrainPred.append( [ trainPred[i, int(classTrainPred[i][0])] ]  )
            #v = int( classifierOutput["trainY"][i]) # Caso Ideal
            v = int( classifierOutput["trainPred"][i]) # Prediccion 'hard'
            ntrainPred.append( [ trainPred[i, v] ]  )
            if classTrainDate[i] != dateTrain[i]:
                print("!!!",True)
        ntrainPred = np.array(ntrainPred)
        
        ntestPred = []
        for i in range(len(testPred)):
            #ntestPred.append( [ testPred[i, int(classTestPred[i][0])] ]  )
            #v=int(classifierOutput["testY"][i]) # Caso Ideal
            v=int(classifierOutput["testPred"][i]) # Prediccion 'hard'
            ntestPred.append( [ testPred[i, v] ]   )
            if classTestDate[i] != dateTest[i]:
                print("!!!,test",True)
        ntestPred = np.array(ntestPred)
    elif PRED_TYPE == "soft":
        ntrainPred = []
        for i in range(len(trainPred)):
            upperProb = float( classifierOutput["trainPred"][i]) # Prediccion 'soft'
            lowerProb = 1-upperProb
            ntrainPred.append( [ trainPred[i, 0]*lowerProb + trainPred[i, 1]*upperProb  ]  )
            if classTrainDate[i] != dateTrain[i]:
                print("!!!",True)
        ntrainPred = np.array(ntrainPred)
        
        ntestPred = []
        for i in range(len(testPred)):
            upperProb = float( classifierOutput["testPred"][i]) # Prediccion 'soft'
            lowerProb = 1-upperProb
            ntestPred.append( [ testPred[i, 0]*lowerProb + testPred[i, 1]*upperProb  ]  )
            if classTestDate[i] != dateTest[i]:
                print("!!!",True)
        ntestPred = np.array(ntestPred)
    
    #print(ntrainPred)
    
    
    # Invert Predictions
    ntrainPred = Yscaler.inverse_transform(ntrainPred)
    trainYinv = Yscaler.inverse_transform(trainY[:, 0])
    
    ntestPred = Yscaler.inverse_transform(ntestPred)
    testYinv = Yscaler.inverse_transform(testY[:, 0])
    
    #print("###", ntrainPred[:5])
    #print("###", trainYinv[:5])
    
    #for x,y,z,zz in zip(trainYinv,classifierOutput["trainY"], dateTrain, classifierOutput["dateTrain"]):
    #    print(z,zz,x,y, x>=THETA, ((x >= THETA) != y)*999999999999)
        
    
    
    # calculate root mean squared error
    print("Calculando Error")
    trainScore = math.sqrt(mean_squared_error(trainYinv, ntrainPred))
    print('    Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testYinv, ntestPred))
    print('    Test Score: %.2f RMSE' % (testScore))
    
    print("Calculando RMSEat%d"%THETA)
    trainScore = RMSEat(THETA,trainYinv, ntrainPred)
    print('    Train Score: %.2f RMSEat%.2f' % (trainScore,THETA))
    testScore = RMSEat(THETA,testYinv, ntestPred)
    print('    Test Score: %.2f RMSEat%.2f' % (testScore, THETA))
    
    '''
    print("mean trainY <, >, mean:", np.mean(trainYinv[trainYinv<THETA], axis=0), np.mean(trainYinv[trainYinv>=THETA], axis=0),np.mean(trainYinv, axis=0) )
    v = internalConfig["classifierOutput"]["trainY"].astype(dtype=bool).flatten()
    v1 = (1-internalConfig["classifierOutput"]["trainY"]).astype(dtype=bool).flatten()
    t= Yscaler.inverse_transform(trainPred)
    print(trainPred.shape, t.shape, v.shape, v1.shape)
    print("mean trainP <, >, mean:",np.mean(t[v1,0], axis=0), np.mean(t[v,1], axis=0),np.mean(t[:,2], axis=0))
    print("mean trainPred <, >, mean:",np.mean(Yscaler.inverse_transform(trainPred), axis=0))
    print("mean nTrainPred:",np.mean(ntrainPred, axis=0))
    print("====")
    print("mean testY <, >, mean:", np.mean(testYinv[testYinv<THETA], axis=0), np.mean(testYinv[testYinv>=THETA], axis=0),np.mean(testYinv, axis=0) )
    v = internalConfig["classifierOutput"]["testY"].astype(dtype=bool).flatten()
    v1 = (1-internalConfig["classifierOutput"]["testY"]).astype(dtype=bool).flatten()
    t= Yscaler.inverse_transform(testPred)
    print(testPred.shape, t.shape, v.shape, v1.shape)
    print("mean testP <, >, mean:",np.mean(t[v1,0], axis=0), np.mean(t[v,1], axis=0),np.mean(t[:,2], axis=0))
    print("mean testPred <, >, mean:",np.mean(Yscaler.inverse_transform(testPred), axis=0))
    print("mean nTestPred:",np.mean(ntestPred, axis=0))
    '''
    
    
    # PLOT
    print("Graficando...")
    df1 = join_index(dateTrain,trainYinv, "Y")
    df2 = join_index(dateTrain,ntrainPred, "f train")
    
    df3 = join_index(dateTest,testYinv, "Y")
    df4 = join_index(dateTest,ntestPred, "f test")
    
    ydf = pd.concat([df1, df3], axis=0)
    df = pd.concat( [ydf, df4,df2] ,axis=1)
    
    #traindf = pd.concat([df1,df2], axis=1)
    #testdf = pd.concat([df3,df4], axis=1)
    
    #df = pd.concat([traindf,testdf])
    
    #fecha_test = testdf.index[0]
    #df.asfreq("D").iplot(title = "%s - %s"%(TARGET, LAYERS), vline=[fecha_test], hline=[THETA])
    df.asfreq("D").iplot(title = "%s - %s"%(TARGET, LAYERS), hline=[THETA])
    
    df = df.asfreq("D")
    
    
    


# In[ ]:


a = np.array([[1,2],[3,4],[5,6]])
f = [True,False,True]
a.flatten()


# ## LSTM_2YModel

# In[ ]:


def LSTM_2YModel(internalConfig, Config):
    ic = internalConfig
    
    TIMESTEP = Config['TIMESTEP']
    TIMEDIST = Config["TIMEDIST"]
    
    #if TIMEDIST == True:
    #    ic["trainY"] = ic["trainY"].reshape(-1, TIMESTEP, y_len )
    #    ic["testY"] = ic["testY"].reshape(-1, TIMESTEP, y_len )
    #else:
    #    ic["trainY"] = ic["trainY"][:,-1]
    #    ic["testY"] = ic["testY"][:,-1]
    
    
    OVERWRITE_MODEL = Config["OVERWRITE_MODEL"]
    TARGET = Config["TARGET"]
    BATCH_SIZE = Config["BATCH_SIZE"]
    THETA = Config["THETA"]
    LAYERS = Config["LAYERS"]
    DROP_RATE = Config["DROP_RATE"]
    EPOCHS = Config["EPOCHS"]
    LOSS = Config["LOSS"]
    f_len = ic["f_len"]
    trainX = ic["trainX"]
    trainY = ic["trainY"]
    testX = ic["testX"]
    testY = ic["testY"]
    y_len = ic["y_len"]
    
    Yscaler = ic["scalers"]["Yscaler"]
    scaled_theta = Yscaler.transform([[ THETA ]])
    
    FILE_NAME = create_filename(ic, Config)
    
    print("###", trainY.shape)
    print("###", testY.shape)
    
    try:
        if OVERWRITE_MODEL == False:
            print("Loading Model...")
            classicModel = load_model(FILE_NAME, custom_objects = {'<lambda>': lambda y,f: loss2YatTHETA(scaled_theta,y,f)})
            print(FILE_NAME + " loaded =)")
            print("LISTO")
        else:
            print("Reentrenando "+ FILE_NAME)
            load_model("noexisto_nijamas_existire.h5")
    except:
        print("batch_input_shape:(%d,%d,%d)"%(BATCH_SIZE,TIMESTEP,f_len))
        classicModel = Sequential()
        
        for Neurons in LAYERS[:-1]:
            classicModel.add(LSTM(Neurons, activation="sigmoid", input_shape=(TIMESTEP, f_len), return_sequences=True))
            classicModel.add(Dropout(rate=DROP_RATE))
        
        if TIMEDIST == True:
            classicModel.add(LSTM(LAYERS[-1], activation="sigmoid", input_shape=(TIMESTEP, f_len), return_sequences=True))
            classicModel.add(Dropout(rate=DROP_RATE))
            classicModel.add( TimeDistributed( Dense( y_len, activation="linear") ) )
            print("JEJEJE")
        else:
            classicModel.add(LSTM(LAYERS[-1], activation="sigmoid", input_shape=(TIMESTEP, f_len), return_sequences=False))
            classicModel.add(Dropout(rate=DROP_RATE))
            classicModel.add(Dense( trainY.shape[1], activation="linear"))
        
        #qModel = Sequential()
        #for Neurons in LAYERS[:-1]:
        #    qModel.add(LSTM(Neurons, input_shape=(TIMESTEP, f_len), return_sequences=True))
        #    qModel.add(Dropout(rate=DROP_RATE))
        #
        #if TIMEDIST == True:
        #    pass
        #else:
        #    qModel.add(LSTM(LAYERS[-1], input_shape=(TIMESTEP, f_len), return_sequences=False))
        #    qModel.add(Dropout(rate=DROP_RATE))
        #    qModel.add(Dense( y_len + len(QUANTILES) ))
        #    qModel.compile(loss=lambda y,f: quantil_loss(QUANTILES, y_len,y,f), optimizer='adam')

        classicModel.compile(loss=lambda y,f: loss2YatTHETA(scaled_theta,y,f), optimizer='adam')
        #classicModel.compile(loss='mean_squared_error', optimizer='adam')
        
        classicModel.fit(trainX, trainY, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, shuffle=False)

        #classicModel.fit(trainX, trainY, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2)
        classicModel.save(FILE_NAME)
        print(FILE_NAME + " saved. = )")
    
    return classicModel, FILE_NAME
    #ic["model"] = classicModel
    #ic["file_name"] = FILE_NAME


# ## Pruebas - cuantílica

# In[ ]:


STATION = "Las_Condes"
qConfig = {
                        "SCALER" : preprocessing.StandardScaler,
                      "AGREGADOS":[],
                         "TARGET": "O3",
                       "PRECALC" : precalcular_agregados(STATION),
                          "THETA":1,
                         "FUTURE":1, #must be 1
                           "PAST":False, #must be False
                        
                 "FIXED_FEATURES":[],
                            "CUT":0.26,
                            "BAN":["countEC","EC","O3btTHETA"],
                        
                      "TIMESTEP" : 14,
                        "OVERLAP":True,
                     
                       "SHUFFLE" : False,
                     "BATCH_SIZE": 16,
                       "TRAINPCT":0.85,
    
                "OVERWRITE_MODEL":False,
                    "MODEL_NAME" : "qModel",
                      "QUANTILES":[0.5],
                        "LAYERS" : [3],
                      "DROP_RATE": 0.4,
                        "EPOCHS" : 250,
                      "TIMEDIST" : False,  #Sin implementar
                            "WW" : None,
    
        }
#Qoutput = quantileLSTM(qConfig)


# In[ ]:


#WW = Qoutput["model"].get_weights()


# In[ ]:


'''
trainTemp, trainDate = Qoutput["trainPred"][:,0], Qoutput["dateTrain"]
testTemp, testDate = Qoutput["testPred"][:,0], Qoutput["dateTest"]
predTemp = np.hstack([trainTemp,testTemp])
dateTemp = np.hstack([trainDate,testDate])
new_column = join_index(dateTemp,predTemp,"O3pred")

adata = Qoutput["data"].copy()
columns = adata.columns
data = Qoutput["complete_dataset"].copy()
#new_column
data["O3"]=Qoutput["scalers"]["Yscaler"].inverse_transform(data["O3"])
data = pd.concat([data,new_column],axis=1)
#data["O3pred"] = new_column
#data = data[["O3","O3pred"]]
#data = data[list(columns)+["O3pred"]]
#data["O3pred"] = data["O3pred"].shift(-1) #EL SHIFT se debe hacer para el cuando se considere como TARGET Y

#print(data["O3pred"].describe())
#print(data)
#print(data["O3pred"])
#print(data2["O3pred"].describe())

temp = data.copy()
#data2 = data.dropna()
#print(data.describe())

print(data["O3"].describe())
data2 = data.asfreq("D")
print( np.mean(data[columns].dropna().index == data2[columns].dropna().index))
print(data2["O3"].describe())
print(len(data))
a = data.apply( lambda x: float("188888") if np.isnan(x["O3pred"]) else (x["O3"] > x["O3pred"])*1, axis=1 )
print(a.describe())
#data["O3btPred"] = (data["O3"] > data["O3pred"])*1

data["O3btPred"] = a
data = data[list(columns)+["O3btPred"]]
data = data.dropna()
data["y"] = data["O3btPred"].shift(1)
#data = data.dropna().drop("O3btPred",axis=1)
#data = data.drop("O3btPred", axis=1).dropna()
print(data[["O3","y"]])
print(adata[["O3","y"]])
print( sum(data.index == adata.index))

# FALTA HACER QUE LOS COINCIDAN BIEN LOS DATOS, SIN LUGAR A ERROR
# CONSIDERAR QUE AL HACER EL SHIFT LOS DIAS DEBE ESTAR CORRELATIVOS
#
#


#
#temp = temp.dropna()
##data["O3btPred"] = temp["O3btPred"]
##data[["O3","O3pred","O3btPred"]]
##adata["O3btPred"] = data["O3btPred"]
#adata["O3btPred"].fillna(value=adata["y"])
#adata["O3btPred"].describe()
#print(random.choice([0.,1.]) if np.isnan(123) else 123 )
#print(data["O3btPred"].describe())
##p=adata["O3btPred"].describe()[1]
##adata["O3btPred"] = adata["O3btPred"].apply( lambda x: np.random.choice([0,1],p=[1-p,p]) if np.isnan(x) else x )
#print(adata["O3btPred"].describe())
#print(adata["KK"].describe())

'''


# In[ ]:


'''
trainPred, trainY, dateTrain = Qoutput["trainPred"][:,0].copy(), Qoutput["trainYtrue"][:,0].copy() , Qoutput["dateTrain"].copy()
testPred, testY, dateTest = Qoutput["testPred"][:,0].copy(), Qoutput["testYtrue"][:,0].copy() , Qoutput["dateTest"].copy()

trainX, testX = Qoutput["trainX"], Qoutput["testX"]

trainX.shape, trainPred.shape, trainY.shape
for i in range(len(trainY)):
    if trainY[i] > trainPred[i]:
        trainY[i] = 1.
    else:
        trainY[i] = 0.
trainY = trainY[:, None]

for i in range(len(testY)):
    if testY[i] > testPred[i]:
        testY[i] = 1.
    else:
        testY[i] = 0.
testY = testY[:, None]

Qoutput["trainY"] = trainY
Qoutput["testY"] = testY

'''


# ## Pruebas - clasificadora

# In[ ]:


STATION = "Las_Condes"
ClassifierLSTMconfig={
                    # import_merge_and_scale()
                    "SCALER" : preprocessing.StandardScaler,
                 "AGREGADOS" : [],#["O3","TEMP","WS","RH"],    #["ALL"] #Horas de los maximos que se quieren agregar.
                   "PRECALC" : precalcular_agregados(STATION),
                    "TARGET" : "O3btTHETA",
                     "THETA" : 1,
                    "FUTURE" : 1,
                      "PAST" : False,
                    
                    # select_features()
            "FIXED_FEATURES" : [], # empty list means that the CUT will be used
                       "CUT" : 0.26,
                       "BAN" : ["countEC","EC","O3btTHETA"],#["countEC", "EC"],# []
                    
                    #make_examples()
              "USE_EXAMPLES" : "DUMMY",#Qoutput,
                  "TIMESTEP" : 14,
                   "OVERLAP" : True,
                    
                    #make_traintest()
                   "SHUFFLE" : False,
                "BATCH_SIZE" : 16,
                  "TRAINPCT" : 0.85,
                    
                    #myLSTM()
           "OVERWRITE_MODEL" : True,
                "MODEL_NAME" : "ClassifierO3btTHETAModel",
                    "LAYERS" : [1],
                    "EPOCHS" : 100,
                 "DROP_RATE" : 0.4,
                  "TIMEDIST" : False,  #SIN IMPLEMENTAR
    
                    # Y 
                 "PRED_TYPE" : "hard",
                    }
#ClassifierOutput = ClassifierLSTM( ClassifierLSTMconfig )


# In[ ]:


#co = ClassifierOutput
#ClassPredict(co, ClassifierLSTMconfig)


# In[ ]:


#print((pd.DataFrame(ClassifierOutput["trainY"])).describe())
#print((pd.DataFrame(ClassifierOutput["testY"])).describe())


# ## Pruebas - LSTM

# In[ ]:


STATION = "Las_Condes"
LSTM2Yconfig = {
         "CLASSIFIER_CONFIG" : ClassifierLSTMconfig,
                    # import_merge_and_scale()
                    "SCALER" : preprocessing.StandardScaler,
                 "AGREGADOS" : [],#["O3","TEMP","WS","RH"],    #["ALL"] #Horas de los maximos que se quieren agregar.
                    "TARGET" : "O3",
                   "PRECALC" : precalcular_agregados(STATION),
                     "THETA" : 92,
                    "FUTURE" : 1,
                      "PAST" : False,
                    
                    # select_features()
            "FIXED_FEATURES" : [], # empty list means that the CUT will be used
                       "CUT" : 0.26,
                       "BAN" : [],#["countEC", "EC"],# []
                    
                    #make_examples()
                  "TIMESTEP" : 2,
                   "OVERLAP" : True,
                    
                    #make_traintest()
                   "SHUFFLE" : True,
                "BATCH_SIZE" : 16,
                   "TRAINPCT":0.85,
                    
                    #myLSTM()
           "OVERWRITE_MODEL" : False,
                "MODEL_NAME" : "LSTM2Ymodel",
                    "LAYERS" : [3],
                  "DROP_RATE": 0.4,
                    "EPOCHS" : 100,
                  "TIMEDIST" : False,   #Sin Implementar
                      "LOSS" : "", # "" or "atTHETA"
         
    
                    #Y to calc Error. For Test Only.
                 "PRED_TYPE" : "soft", # sobreescribe PRED_TYPE de la clasificadora
                        "Yx" : 0    # DEFAULT 0
                }
#LSTM_2Youtput = LSTM_2Y(LSTM2Yconfig)


# In[ ]:


#l2y=LSTM_2Youtput
#LSTM_2YPredict(l2y,LSTM2Yconfig)


# In[ ]:


#a=LSTM_2Youtput
##print(a["dateExamples"])
##print(a["classifierOutput"]["dateExamples"])
#b = a["complete_dataset"].copy()
#print(a["complete_dataset"][["O3","O3btTHETA"]])
#print(a["complete_dataset"]["O3"].dropna() > 0)
#b["jeje"] = b["O3"].dropna() > 0
#b[["O3","jeje"]].describe()
#a["data"]["O3"]["2003-12-14":"2003-12-14"][0]
##print(a["classifierOutput"]["complete_dataset"].describe())


# In[ ]:


#a = pd.DataFrame([1,2,3,4,5,6,7,8,9])
#b = pd.DataFrame([10,11,12,13])
##a[1] = b
#pd.concat([a,b], axis=1)


# # Quantile Regression LSTM

# ## make_qY
# Replica los array con las etiquetas Y las veces necesarias para entrenar el valor promedio y los cuantiles.

# In[ ]:


# Modify trainY and testY for Quantile Regression
def make_qY(internalConfig, Config):
    trainY = internalConfig["trainY"]
    validY = internalConfig["validY"]
    testY = internalConfig["testY"]
    QUANTILES = Config["QUANTILES"]
    TIMEDIST = Config["QUANTILES"]
    
    print("Making Y for quantiles")
    ## Si hay problemas hacer train[:,None]
    #print(trainY.shape)
    #print(trainY[:,None].shape)
    #trainYq = trainY[:,None]
    #testYq = testY[:,None]
    
    if TIMEDIST == True:
        pass
    else:
        trainYq = trainY[:, -1, :]
        validYq = validY[:, -1, :]
        testYq = testY[:, -1, :]
        
        trainY = trainY[:, -1, 0, None]
        validY = validY[:, -1, 0, None]
        testY = testY[:, -1, 0, None]
        
    
        for i in range(len(QUANTILES)):
            #trainYq = np.hstack([trainYq, trainY[:,None]])
            #testYq = np.hstack([testYq, testY[:,None]])
            trainYq = np.hstack([trainYq, trainY[:]])
            validYq = np.hstack([validYq, validY[:]])
            testYq = np.hstack([testYq, testY[:]])
    
    print("    trainYq.shape, ", trainYq.shape)
    print("    testYq.shape, ", testYq.shape)
    return trainYq, validYq, testYq


# ## quantil_loss
# Función de perdida para entrenar los cuantiles.  
# El valor total de la función contempla la salida del promedio y la de cada cuantil.

# In[ ]:


# Loss Function
def quantil_loss_old(quantiles, ylen, ytrue, ypred):
    loss = 0
    for i in range(ylen):
        loss += K.mean(K.square(ytrue[:, i]-ypred[:, i]), axis=-1)
    
    #loss = K.mean(K.square(ytrue[:, 0]-ypred[:, 0]), axis=-1)
        
    for k in range(len(quantiles)):
        q = quantiles[k]
        e = (ytrue[:, ylen+k]-ypred[:, ylen+k])
        loss += K.mean(q*e + K.clip(-e, K.epsilon(), np.inf), axis=-1)
    return loss


# In[ ]:


# Loss Function
def quantil_loss(quantiles, ylen, qlen, ytrue, ypred):
    loss = 0
    
    for i in range(ylen):
        loss += K.mean(K.square(ytrue[:,0] - ypred[:, i]), axis=-1)
    
    #loss = K.mean(K.square(ytrue[:, 0]-ypred[:, 0]), axis=-1)
        
    for k in range(qlen):
        q = quantiles[k]
        e = ( ytrue[:,0] - ypred[:, ylen+k])
        loss += K.mean(q*e + K.clip(-e, K.epsilon(), np.inf), axis=-1)
    return loss


# In[ ]:


# Loss Function
def meanquantil_loss(quantiles, ylen,qlen, ytrue, ypred):
    loss = 0
    
    for i in range(ylen):
        loss += K.mean(K.square(ytrue[:,0] - ypred[:, i]), axis=-1)
    
    #loss = K.mean(K.square(ytrue[:, 0]-ypred[:, 0]), axis=-1)
    for k in range(qlen):
        q = quantiles[k]
        e = ( ytrue[:,0] - ypred[:, ylen+k])
        loss += K.mean(q*e + K.clip(-e, K.epsilon(), np.inf), axis=-1)
    return loss/(ylen+qlen)


# In[ ]:


# Loss Function
def meanquantil_loss2(quantiles, ylen,qlen, ytrue, ypred):
    loss = 0
    
    for i in range(ylen):
        loss += K.mean(K.square(ytrue[:,0] - ypred[:, i]), axis=-1)
    
    #loss = K.mean(K.square(ytrue[:, 0]-ypred[:, 0]), axis=-1)
    qloss = 0
    for k in range(qlen):
        q = quantiles[k]
        e = ( ytrue[:,0] - ypred[:, ylen+k])
        qloss += K.mean(q*e + K.clip(-e, K.epsilon(), np.inf), axis=-1)
    qloss = qloss/qlen
    return loss + qloss


# ## Qmodel
# Función que entrena el modelo para predecir los cuantiles en el tiempo T+1. 

# In[ ]:


# Model
# [4], [100], [4,4], [100,100], [50,50], [50, 50, 10], [30, 30, 30, 30], [4, 4, 4, 4]

def Qmodel(internalConfig, Config):#, trainX, trainY, ylen, FUTURE, PAST, TARGET, BATCH_SIZE, TIMESTEP, FEATURES, TRAINPCT, OVERWRITE_MODEL, MODEL_NAME, QUANTILES, LAYERS, EPOCHS, TIMEDIST):
    SEED = Config['SEED']
    np.random.seed(SEED)
    set_random_seed(SEED*SEED)
    
    OVERWRITE_MODEL = Config["OVERWRITE_MODEL"]
    LAYERS = Config["LAYERS"]
    TIMESTEP = Config["TIMESTEP"]
    BATCH_SIZE = Config["BATCH_SIZE"]
    FOLDS_TVT = Config["FOLDS_TVT"]
    EPOCHS = Config["EPOCHS"]
    PATIENCE = Config["PATIENCE"]
    DROP_RATE = Config["DROP_RATE"]
    QUANTILES = Config["QUANTILES"]
    TIMEDIST = Config["TIMEDIST"]
    LOSS = Config["QLOSS"]
    y_len = internalConfig["y_len"]
    f_len = internalConfig["f_len"]
    
    #trainX = internalConfig["trainX"]
    #trainY = internalConfig["trainY"]
    #validX = internalConfig["validX"]
    #validY = internalConfig["validY"]
    
    if FOLDS_TVT == False:
        list_trainX = [ internalConfig["trainX"] ]
        list_trainY = [ internalConfig["trainY"] ]
        list_validX = [ internalConfig["validX"] ]
        list_validY = [ internalConfig["validY"] ]
    else:
        list_trainX = internalConfig['list_trainX']
        list_trainY = internalConfig['list_trainY']
        list_validX = internalConfig['list_validX']
        list_validY = internalConfig['list_validY']
    
    
    LAYERS.reverse()
    DROP_RATE.reverse()
    losses = []
    list_models = []
    for fold in range(0,len(list_trainX)):
        internalConfig['fold'] = fold + 1
        print("Using Fold:", fold)
        trainX = list_trainX[fold]
        trainY = list_trainY[fold]
        validX = list_validX[fold]
        validY = list_validY[fold]
        
    
        f_len = trainX.shape[-1]
        
        FILE_NAME = create_filename(internalConfig, Config)
        
        
        try:
            print("Model File Name: ",FILE_NAME)
            if OVERWRITE_MODEL == False:
                print("Loading Model...")
                qModel = load_model(FILE_NAME, custom_objects = {'<lambda>': lambda y,f: LOSS(QUANTILES,y_len,len(QUANTILES),y,f)})
                list_models.append(qModel)
                #qModel.summary()
                print(FILE_NAME + " Loaded =)")
            else:
                print("Reentrenando "+ FILE_NAME)
                load_model("noexisto_nijamas_existire.hacheCinco")
        except:
        #        qModel = Sequential()
        #        for Neurons in LAYERS[:-1]:
        #            qModel.add(LSTM(Neurons, batch_input_shape=(BATCH_SIZE, TIMESTEP, f_len), stateful=True, return_sequences=True))
        #        qModel.add(LSTM(LAYERS[-1], batch_input_shape=(BATCH_SIZE, TIMESTEP, f_len), stateful=True))
        #        qModel.add(Dense( 1+len(QUANTILES) ))
        #        #qModel.add(Dense( 1 ))
        #        qModel.summary()
        #        qModel.compile(loss=lambda y,f: quantil_loss(QUANTILES,y,f), optimizer='adam')
        #        #qModel.compile(loss='mean_squared_error', optimizer='adam')
        #        for i in range(EPOCHS):
        #            print("%i/%i"%(i+1,EPOCHS))
        #            qModel.fit(trainX, trainY, epochs=1, batch_size=BATCH_SIZE, verbose=1, shuffle=False)
        #            qModel.reset_states()
           # print("WWW", trainY.shape)
           # print("WWW", trainY)
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=PATIENCE)
            qModel = Sequential()
            if len(LAYERS) == 2:
                qModel.add(LSTM(LAYERS[1], activation="sigmoid", input_shape=(TIMESTEP, f_len), return_sequences=True))
                qModel.add(Dropout(rate=DROP_RATE[1]))
            
            if TIMEDIST == True:
                pass
            else:
                qModel.add(LSTM(LAYERS[0], activation="sigmoid", input_shape=(TIMESTEP, f_len), return_sequences=False))
                qModel.add(Dropout(rate=DROP_RATE[0]))
                qModel.add(Dense( y_len + len(QUANTILES), activation="linear" ))
            
            #qModel.summary()
            qModel.compile(loss=lambda y,f: LOSS(QUANTILES, y_len,len(QUANTILES),y,f), optimizer='adam')
            
            qModel.fit(trainX, trainY, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(validX,validY), callbacks=[es], verbose=1)
            
            qModel.save(FILE_NAME)
            print(FILE_NAME + " Saved =)")
            
            list_models.append(qModel)
    
    return list_models, FILE_NAME


# ## Qprediction
# Función que realiza predicciones para el modelo entrenado mediante la función Qmodel.  

# In[ ]:


def Qprediction(internalConfig, Config):
    QUANTILES = Config["QUANTILES"]
    BATCH_SIZE = Config["BATCH_SIZE"]
    THETA = Config["THETA"]
    moreTHETA = Config["moreTHETA"]
    TIMEDIST = Config["TIMEDIST"]
    FOLDS_TVT = Config["FOLDS_TVT"]
    GRAPH = Config["GRAPH"]
    #trainX = internalConfig["trainX"]
    #trainY = internalConfig["trainY"]
    #validX = internalConfig["validX"]
    #validY = internalConfig["validY"]
    #testX = internalConfig["testX"]
    #testY = internalConfig["testY"]
    Yscaler = internalConfig["scalers"]["Yscaler"]
    h24scaler = internalConfig["scalers"]["h24scaler"]
    list_models = internalConfig["list_models"]
    
    
    #WW = Config["WW"]
    
    #if WW != None:
    #    qModel.set_weights(WW)
    
    if FOLDS_TVT == False:
        list_trainX = [ internalConfig[   "trainX"] ]
        list_trainY = [ internalConfig[   "trainY"] ]
        list_validX = [ internalConfig[   "validX"] ]
        list_validY = [ internalConfig[   "validY"] ]
        list_testX  = [ internalConfig[   "testX" ] ]
        list_testY  = [ internalConfig[   "testY" ] ]
        list_dateTrain = [ internalConfig[ "dateTrain"] ]
        list_dateValid = [ internalConfig[ "dateValid"] ]
        list_dateTest  = [ internalConfig[ "dateTest"] ] 
        
    else:
        list_trainX = internalConfig[   "list_trainX"]
        list_trainY = internalConfig[   "list_trainY"]
        list_validX = internalConfig[   "list_validX"]
        list_validY = internalConfig[   "list_validY"]
        list_testX  = internalConfig[   "list_testX"]
        list_testY  = internalConfig[   "list_testY"]
        list_dateTrain = internalConfig["list_dateTrain"]
        list_dateValid = internalConfig["list_dateValid"]
        list_dateTest  = internalConfig["list_dateTest"]
    
    # RMSE
    trainRMSE = []
    validRMSE = []
    testRMSE  = []
    
    # MAE
    trainMAE = []
    validMAE = []
    testMAE  = []
    
    scores_detail = {
                "train": {'quantity':[]},
                "valid": {'quantity':[]},
                "test" : {'quantity':[]},
                }
    
    # quantile metrics
    trainQmetricAll =[]
    validQmetricAll = []
    testQmetricAll  = []
    trainQmetricAtTHETA =[]
    validQmetricAtTHETA = []
    testQmetricAtTHETA  = []
    
    # Interval Coverage
    trainIC = []
    validIC = []
    testIC  = []
    
    
    print("Qprediction ...")
    for fold in range(0,len(list_trainX)):
        trainX = list_trainX[fold]
        trainY = list_trainY[fold]
        validX = list_validX[fold]
        validY = list_validY[fold]
        testX  =  list_testX[fold]
        testY  =  list_testY[fold]
        dateTrain = list_dateTrain[fold]
        dateValid = list_dateValid[fold]
        dateTest  = list_dateTest[fold] 
    
        qModel = list_models[fold]
    
        # Predictions
        
        qlen = len(QUANTILES)
        trainPred = qModel.predict(trainX)
        qModel.reset_states()
        validPred = qModel.predict(validX)
        qModel.reset_states()
        testPred = qModel.predict(testX)
        qModel.reset_states()
        
        if TIMEDIST == True:
            pass
        else:
            #print("trainPred.shape, ",trainPred.shape)
            #print("trainY.shape, ",trainY.shape)
            #print("validPred.shape, ",validPred.shape)
            #print("validY.shape, ",validY.shape)
            #print("testPred.shape, ",testPred.shape)
            #print("testY.shape, ",testY.shape)
            #Reduces shape to ( examples, y + QUANTILES). Discarding y0, y2, y-1, etc...
            trainPred = np.hstack( [trainPred[:,0,None], trainPred[:, -qlen:] ] )
            validPred = np.hstack( [validPred[:,0,None], validPred[:, -qlen:] ] )
            testPred  = np.hstack( [testPred[:,0,None],  testPred[:, -qlen:] ] )
            #trainY = np.hstack( [trainY[:,0,None], trainY[:, -qlen:] ] )
            #testY = np.hstack( [testY[:,0,None], testY[:, -qlen:] ] )
            #print("trainPred.shape, ",trainPred.shape)
            #print("trainY.shape, ",trainY.shape)
            #print("testPred.shape, ",testPred.shape)
            #print("testY.shape, ",testY.shape)
        
        
        # Inverse Transform
        finalTrainPredq = Yscaler.inverse_transform(trainPred[:,0, None])
        #print(trainY.shape)
        #print(trainY[:,0,None].shape)
        finalTrainY = Yscaler.inverse_transform(trainY)
        #print(finalTrainY.shape)
        finalValidPredq = Yscaler.inverse_transform(validPred[:,0, None])
        finalValidY = Yscaler.inverse_transform(validY)
        finalTestPredq = Yscaler.inverse_transform(testPred[:,0,None])
        finalTestY = Yscaler.inverse_transform(testY)
        
        
        # stats
        num_train = len(trainX)
        num_valid = len(validX)
        num_test = len(testX)
        
        num_THETA_train = np.sum(finalTrainY >= THETA)
        num_THETA_valid = np.sum(finalValidY >= THETA)
        num_THETA_test = np.sum(finalTestY >= THETA)
        
        scores_detail["train"]['quantity'].append( (num_train, num_THETA_train) )
        scores_detail["valid"]['quantity'].append( (num_valid, num_THETA_valid) )
        scores_detail["test"]['quantity'].append(  (num_test, num_THETA_test)   )
        
        
        # calculate root mean squared error
        yt, yp = finalTrainY, finalTrainPredq
        scores = RMSE(yt, yp, THETA=THETA, ALL=True)
        scores = [scores["RMSE"], scores["RMSPE"], scores["RMSEat"], scores["RMSPEat"] ]
        for t in moreTHETA:
            scores.append( RMSE(yt, yp, THETA=t) )
            scores.append( RMSE(yt, yp, THETA=t, norm=True) )
        trainRMSE.append( scores )
        yt, yp = finalValidY, finalValidPredq
        scores = RMSE(yt, yp, THETA=THETA, ALL=True)
        scores = [scores["RMSE"], scores["RMSPE"], scores["RMSEat"], scores["RMSPEat"] ]
        for t in moreTHETA:
            scores.append( RMSE(yt, yp, THETA=t) )
            scores.append( RMSE(yt, yp, THETA=t, norm=True) )
        validRMSE.append( scores )
        yt, yp = finalTestY, finalTestPredq
        scores = RMSE(yt, yp, THETA=THETA, ALL=True)
        scores = [scores["RMSE"], scores["RMSPE"], scores["RMSEat"], scores["RMSPEat"] ]
        for t in moreTHETA:
            scores.append( RMSE(yt, yp, THETA=t) )
            scores.append( RMSE(yt, yp, THETA=t, norm=True) )
        testRMSE.append( scores )
        a,b = trainRMSE[-1][0:2]
        c,d = validRMSE[-1][0:2]
        e,f = testRMSE[-1][0:2]
        print("fold %s score (RMSE,RMSPE): (%.2f, %.2f) (%.2f, %.2f) (%.2f, %.2f)"%(fold, a,b,c,d,e,f))
        
        '''
        scores = RMSE(finalTrainY, finalTrainPredq, THETA=THETA, ALL=True)
        trainRMSE.append( [scores["RMSE"], scores["RMSPE"], scores["RMSEat"], scores["RMSPEat"]] )
        scores = RMSE(finalValidY, finalValidPredq, THETA=THETA, ALL=True)
        validRMSE.append( [scores["RMSE"], scores["RMSPE"], scores["RMSEat"], scores["RMSPEat"]] )
        scores = RMSE(finalTestY, finalTestPredq, THETA=THETA, ALL=True)
        testRMSE.append(  [scores["RMSE"], scores["RMSPE"], scores["RMSEat"], scores["RMSPEat"]] )
        
        trainScore = RMSE(finalTrainY, finalTrainPredq, 61)
        trainRMSE[-1].append( trainScore )
        validScore = RMSE(finalValidY, finalValidPredq, 61)
        validRMSE[-1].append( validScore )
        testScore = RMSE(finalTestY, finalTestPredq, 61)
        testRMSE[-1].append( testScore )
        '''
        
        #MAE
        yt, yp = finalTrainY, finalTrainPredq
        scores = MAE(yt, yp, THETA=THETA, ALL=True)
        scores = [scores["MAE"], scores["MAPE"], scores["MAEat"], scores["MAPEat"] ]
        for t in moreTHETA:
            scores.append( MAE(yt, yp, THETA=t) )
            scores.append( MAE(yt, yp, THETA=t, norm=True) )
        trainMAE.append( scores )
        yt, yp = finalValidY, finalValidPredq
        scores = MAE(yt, yp, THETA=THETA, ALL=True)
        scores = [scores["MAE"], scores["MAPE"], scores["MAEat"], scores["MAPEat"] ]
        for t in moreTHETA:
            scores.append( MAE(yt, yp, THETA=t) )
            scores.append( MAE(yt, yp, THETA=t, norm=True) )
        validMAE.append( scores )
        yt, yp = finalTestY, finalTestPredq
        scores = MAE(yt, yp, THETA=THETA, ALL=True)
        scores = [scores["MAE"], scores["MAPE"], scores["MAEat"], scores["MAPEat"] ]
        for t in moreTHETA:
            scores.append( MAE(yt, yp, THETA=t) )
            scores.append( MAE(yt, yp, THETA=t, norm=True) )
        testMAE.append( scores )
        
        '''
        scores = MAE(finalTrainY, finalTrainPredq, THETA=THETA, ALL=True)
        trainMAE.append( [scores["MAE"], scores["MAPE"], scores["MAEat"], scores["MAPEat"]] )
        scores = MAE(finalValidY, finalValidPredq, THETA=THETA, ALL=True)
        validMAE.append( [scores["MAE"], scores["MAPE"], scores["MAEat"], scores["MAPEat"]] )
        scores = MAE(finalTestY, finalTestPredq, THETA=THETA, ALL=True)
        testMAE.append( [scores["MAE"], scores["MAPE"], scores["MAEat"], scores["MAPEat"]] )
        '''
        
        
        # quantile metric
        trainMean , trainQMlosses = quantile_metrics(finalTrainY, QUANTILES, Yscaler.inverse_transform(trainPred[:,1:]))
        trainQlosses = list(trainQMlosses) + [trainMean]
        trainQmetricAll.append( trainQlosses )
        validMean , validQMlosses = quantile_metrics(finalValidY, QUANTILES, Yscaler.inverse_transform(validPred[:,1:]))
        validQlosses = list(validQMlosses) + [validMean]
        validQmetricAll.append( validQlosses )
        testMean , testQMlosses = quantile_metrics(finalTestY, QUANTILES, Yscaler.inverse_transform(testPred[:,1:]))
        testQlosses = list(testQMlosses) + [testMean]
        testQmetricAll.append( testQlosses )
        
        # Interval Coverage
        bp = np.mean(Yscaler.inverse_transform(trainPred[:,0]))
        nominal_sig = 1.0 - (QUANTILES[-1] - QUANTILES[0])
        
        inf_limit = Yscaler.inverse_transform(trainPred[:,1])
        sup_limit = Yscaler.inverse_transform(trainPred[:,-1])
        MIS, MSIS, cov_prob, lenght = IC_metrics(inf_limit, sup_limit, nominal_sig, finalTrainY, bp)
        trainIC.append( [MIS, MSIS, cov_prob, lenght] )
        
        inf_limit = Yscaler.inverse_transform(validPred[:,1])
        sup_limit = Yscaler.inverse_transform(validPred[:,-1])
        MIS, MSIS, cov_prob, lenght = IC_metrics(inf_limit, sup_limit, nominal_sig, finalValidY, bp)
        validIC.append( [MIS, MSIS, cov_prob, lenght] )
        
        inf_limit = Yscaler.inverse_transform(testPred[:,1])
        sup_limit = Yscaler.inverse_transform(testPred[:,-1])
        MIS, MSIS, cov_prob, lenght = IC_metrics(inf_limit, sup_limit, nominal_sig, finalTestY, bp)
        testIC.append( [MIS, MSIS, cov_prob, lenght] )
        
        
        # Only for graphics
        if GRAPH == True:
            for i in range(len(QUANTILES)):
                temp = Yscaler.inverse_transform(trainPred[:,1+i,None])
                finalTrainPredq = np.hstack([finalTrainPredq, temp])
                #temp = Yscaler.inverse_transform(trainY[:,1+i,None])
                #finalTrainY =np.hstack([finalTrainY, temp])
                finalTrainY = Yscaler.inverse_transform(trainY)
                
                temp = Yscaler.inverse_transform(validPred[:,1+i,None])
                finalValidPredq = np.hstack([finalValidPredq, temp])
                finalValidY = Yscaler.inverse_transform(validY)
                
                temp = Yscaler.inverse_transform(testPred[:,1+i,None])
                finalTestPredq = np.hstack([finalTestPredq, temp])
                #temp = Yscaler.inverse_transform(testY[:,1+i,None])
                #finalTestY = np.hstack([finalTestY, temp])
                finalTestY = Yscaler.inverse_transform(testY)
                
            internalConfig["trainPred"] = finalTrainPredq
            internalConfig["trainYtrue"] = finalTrainY
            internalConfig["validPred"] = finalValidPredq
            internalConfig["validYtrue"] = finalValidY
            internalConfig["testPred"] = finalTestPredq
            internalConfig["testYtrue"] = finalTestY
            internalConfig["last_dateTrain"] = dateTrain
            internalConfig["last_dateValid"] = dateValid
            internalConfig["last_dateTest"]  = dateTest
            
            graph_Qprediction(internalConfig, Config)
        
        
        
    
    #print(finalTrainPredq.shape)
        
    
    print("### LSTM CUANTÍLICA ###")
    labels = ["RMSE", "RMSPE", "RMSEat%s"%THETA, "RMSPEat%s"%THETA]
    for t in moreTHETA:
        labels.append("RMSEat%s"%t)
        labels.append("RMSPEat%s"%t)
    maxlen = max(map(len,labels))
    trainMeans = np.nanmean(trainRMSE, axis=0)
    validMeans = np.nanmean(validRMSE, axis=0)
    testMeans  = np.nanmean(testRMSE, axis=0)
    print("{:>{maxlen}}{:>8}{:>8}{:>8}".format("","Train","Valid","Test",maxlen=maxlen))
    for i in range(len(labels)):
        print( ("{0:>{maxlen}}{1:8.2f}{2:8.2f}{3:8.2f}".format (labels[i], trainMeans[i], validMeans[i], testMeans[i], maxlen=maxlen)) )
        
        scores_detail["train"][labels[i]] = np.array(trainRMSE)[:,i]
        scores_detail["valid"][labels[i]] = np.array(validRMSE)[:,i]
        scores_detail["test"][labels[i]]  = np.array(testRMSE)[:,i]
        
    print("\n")
    '''
    labels = ["RMSE", "RMSPE", "RMSEat%s"%THETA, "RMSPEat%s"%THETA, "RMSEat61"]
    maxlen = max(map(len,labels))
    trainMeans = np.nanmean(trainRMSE, axis=0)
    validMeans = np.nanmean(validRMSE, axis=0)
    testMeans  = np.nanmean(testRMSE, axis=0)
    print("{:>{maxlen}}{:>8}{:>8}{:>8}".format("","Train","Valid","Test",maxlen=maxlen))
    for i in range(len(labels)):
        print( ("{0:>{maxlen}}{1:8.2f}{2:8.2f}{3:8.2f}".format (labels[i], trainMeans[i], validMeans[i], testMeans[i], maxlen=maxlen)) )
    print("\n")
    '''
    
    labels = ["MAE", "MAPE", "MAEat%s"%THETA, "MAPEat%s"%THETA]
    for t in moreTHETA:
        labels.append("RMSEat%s"%t)
        labels.append("RMSPEat%s"%t)
    maxlen = max(map(len,labels))
    trainMeans = np.nanmean(trainMAE, axis=0)
    validMeans = np.nanmean(validMAE, axis=0)
    testMeans  = np.nanmean( testMAE, axis=0)
    print("{:>{maxlen}}{:>8}{:>8}{:>8}".format("","Train","Valid","Test",maxlen=maxlen))
    for i in range(len(labels)):
        print( ("{0:>{maxlen}}{1:8.2f}{2:8.2f}{3:8.2f}".format (labels[i], trainMeans[i], validMeans[i], testMeans[i], maxlen=maxlen)) )
    
        scores_detail["train"][labels[i]] = np.array(trainMAE)[:,i]
        scores_detail["valid"][labels[i]] = np.array(validMAE)[:,i]
        scores_detail["test"][labels[i]]  = np.array(testMAE)[:,i]
    print("\n")
    
    '''
    labels = ["MAE", "MAPE", "MAEat%s"%THETA, "MAPEat%s"%THETA]
    maxlen = max(map(len,labels))
    trainMeans = np.nanmean(trainMAE, axis=0)
    validMeans = np.nanmean(validMAE, axis=0)
    testMeans  = np.nanmean( testMAE, axis=0)
    print("{:>{maxlen}}{:>8}{:>8}{:>8}".format("","Train","Valid","Test",maxlen=maxlen))
    for i in range(len(labels)):
        print( ("{0:>{maxlen}}{1:8.2f}{2:8.2f}{3:8.2f}".format (labels[i], trainMeans[i], validMeans[i], testMeans[i], maxlen=maxlen)) )
    print("\n")
    '''
    
    print("Quantile Metric")
    labels = QUANTILES + ["mean"]
    trainMeans = np.nanmean(trainQmetricAll,axis=0)
    validMeans = np.nanmean(validQmetricAll,axis=0)
    testMeans  = np.nanmean(testQmetricAll,axis=0)
    print("\t\tTrain\tValid\tTest")
    for i in range(len(labels)):
        print("\t%s\t%.2f\t%.2f\t%.2f" % (labels[i],trainMeans[i],validMeans[i],testMeans[i]))
    print("\n")
    
    labels = ["MIS", "MSIS", "cov_prob", "lenght"]
    trainMeans = np.nanmean(trainIC,axis=0)
    validMeans = np.nanmean(validIC,axis=0)
    testMeans  = np.nanmean(testIC ,axis=0)
    print("{:>{maxlen}}{:>8}{:>8}{:>8}".format("","Train","Valid","Test",maxlen=maxlen))
    for i in range(len(labels)):
        print( ("{0:>{maxlen}}{1:8.2f}{2:8.2f}{3:8.2f}".format (labels[i], trainMeans[i], validMeans[i], testMeans[i], maxlen=maxlen)) )
    
    #print("finalTrainPredq.shape, ",finalTrainPredq.shape)
    #print("finalTrainY.shape, ",finalTrainY.shape)
    #print("finalTestPredq.shape, ",finalTestPredq.shape)
    #print("finalTestY.shape, ",finalTestY.shape)
    #return (finalTrainPredq, finalTrainY, finalValidPredq, finalValidY, finalTestPredq, finalTestY)
    return scores_detail


# ## graph_Qprediction
# Grafica las predicciones realizadas por la función Qprediction.

# In[ ]:


def graph_Qprediction(internalConfig, Config):
    QUANTILES = Config["QUANTILES"]
    THETA = Config["THETA"]
    
    finalTrainPredq = internalConfig["trainPred"]
    finalTrainY = internalConfig["trainYtrue"]
    finalValidPredq = internalConfig["validPred"]
    finalValidY = internalConfig["validYtrue"]
    finalTestPredq = internalConfig["testPred"]
    finalTestY = internalConfig["testYtrue"]
    dateTrain = internalConfig["last_dateTrain"]
    dateValid = internalConfig["last_dateValid"]
    dateTest  = internalConfig["last_dateTest"]
    date_index = internalConfig["dateExamples"]
    
    file_name = internalConfig["file_name"]
    dataset = internalConfig["complete_dataset"]
    Yscaler = internalConfig["scalers"]["Yscaler"]
    
    
    # Graph quantile TrainPred
    #forPlot = np.hstack([dateTrain[:,None], finalTrainY[:,0,None] ])
    
    print(dateTrain[:,None].shape)
    print(finalTrainY.shape)
    
    print("first dateTrain:", dateTrain[0])
    print("last dateTrain:", dateTrain[-1])
    print("first dateValid:", dateValid[0])
    print("last dateValid:", dateValid[-1])
    print("first dateTest:", dateTest[0])
    print("last dateTest:", dateTest[-1])
    
    forPlot = np.hstack([dateTrain[:,None], finalTrainY ])
    for c in finalTrainPredq.T[:,:,None]:
        forPlot = np.hstack([forPlot, c])
    #print("forPLot.shape, ", forPlot.shape)
    df = pd.DataFrame(forPlot)
    df.columns = ["fecha", "y", "f"] + list( map(str,QUANTILES) )
    df = df.set_index("fecha")
    #print(df.asfreq("D").iplot(title=FILE_NAME))
    
    #'''
    # Graph quantile validPred
    forPlot = np.hstack([dateValid[:,None], finalValidY ])
    for c in finalValidPredq.T[:,:,None]:
        forPlot = np.hstack([forPlot, c])
    dv = pd.DataFrame(forPlot)
    dv.columns = ["fecha", "y", "f"] + list( map(str,QUANTILES) )
    dv = dv.set_index("fecha")
    #'''
    
    # Graph quantile TestPred
    #forPlot = np.hstack([dateTest[:,None], finalTestY[:,0,None] ])
    forPlot = np.hstack([dateTest[:,None], finalTestY ])
    for c in finalTestPredq.T[:,:,None]:
        forPlot = np.hstack([forPlot, c])
    #print("forPLot.shape, ", forPlot.shape)
    dd = pd.DataFrame(forPlot)
    dd.columns = ["fecha", "y", "f"] + list( map(str,QUANTILES) )
    dd = dd.set_index("fecha")
    #print(dd.asfreq("D").iplot(title=FILE_NAME))
    #print(dd.iplot(title=FILE_NAME))

    #cc = pd.concat([df,dd], axis=0).drop("y",axis=1)
    cc = pd.concat([df,dv,dd], axis=0).drop("y",axis=1)
    i = date_index[0] + np.timedelta64(-1,"D")
    f = date_index[-1] + np.timedelta64(-1,"D")

    allY = dataset["y"][i:f]
    allY.index = allY.index + np.timedelta64(1,"D")
    allY = pd.DataFrame(Yscaler.inverse_transform(allY[:,None]), index=allY.index)
    allY.columns = ["y"]

    hh = pd.concat([allY,cc], axis= 1)

    hh.iplot(title=file_name, vline=[dateValid[0],dateTest[0]], hline=[THETA])
    return hh


# ## quantileLSTM
# Llama a las otras funciones para entrenar un modelo, realizar predicciones y graficar.  
# Devuelve el modelo entrenado y un DataFrame con las predicciones para graficar directamente.

# In[ ]:


def quantileLSTM(Config):#, AGREGADOS, TARGET, THETA, FUTURE, PAST, FEATURES, CUT, BAN, TIMESTEP, OVERLAP, BATCH_SIZE, TRAINPCT, OVERWRITE_MODEL, MODEL_NAME, QUANTILES, LAYERS, EPOCHS, TIMEDIST):
    #np.random.seed(123)
    #set_random_seed(2)
    
    FOLDS_TVT = Config['FOLDS_TVT']
    TIMESTEP = Config['TIMESTEP']
    TIMEDIST = Config['TIMEDIST']
    Config['SHIFT'] = Config['FUTURE'] * -1
    
    #SHIFT = FUTURE*-1
    #PRECALC = precalcular_agregados()
    #dataset, ylabels, Yscaler, h24scaler = import_merge_and_scale(AGREGADOS,TARGET, THETA, PRECALC, SHIFT, PAST)
    
    scalers = {}
    ic = {"scalers": scalers}
    
    complete_dataset, ylabels, Yscaler, h24scaler = import_merge_and_scale(Config, verbose=False)
    ic["complete_dataset"] = complete_dataset
    ic["ylabels"] = ylabels
    scalers['Yscaler'] = Yscaler
    scalers["h24scaler"] = h24scaler
    y_len = len(ic["ylabels"])
    ic["y_len"] = y_len
    
    data, features = select_features(ic, Config)
    ic["data"] = data
    ic["features"] = features
    ic["f_len"] = len(ic["features"])
    
    ic["secuencias"] = obtener_secuencias(ic)
    
    examples, y_examples, dateExamples = make_examples(ic, Config, verbose=False)
    ic["examples"] = examples
    ic["y_examples"] = y_examples
    ic["dateExamples"] = dateExamples
    
    if FOLDS_TVT == False:
        tvtDict = make_traintest(ic, Config, verbose=False)
        for key in tvtDict:
            ic[key] = tvtDict[key]
        
    else:
        list_tvtDict = make_folds_TVT(ic, Config)
        for key in list_tvtDict:
            ic[key] = list_tvtDict[key]
    
    #trainYq, testYq = make_qY(ic, Config)
    #ic["trainYq"] = trainYq
    #ic["validYq"] = validYq
    #ic["testYq"] = testYq

    
    
    #trainYq = trainYq#.reshape(-1,5)
    #print("trainYq.shape, ", trainYq.shape)
    list_models, file_name = Qmodel(ic, Config)
    if FOLDS_TVT == False:
        ic["model"] = list_models[0]
        ic["list_models"] = [ ic["model"] ]
    else:
        ic["list_models"] = list_models
    
    ic["file_name"] = file_name
    
    
    ##ftrp-> TRaining Prediction
    ##ftry-> TRaining Y
    ##ftep-> TEst Prediction
    ##ftey-> TEst Y
    #ftrp, ftry, fvap, fvay, ftep, ftey = Qprediction(ic, Config)
    #ic["trainPred"] = ftrp
    #ic["trainYtrue"] = ftry
    #ic["validPred"] = fvap
    #ic["validYtrue"] = fvay
    #ic["testPred"] = ftep
    #ic["testYtrue"] = ftey
    
    detail = Qprediction(ic, Config)
    
    #if FOLDS_TVT == False:
    #Qdf = graph_Qprediction(ic, Config)
    #ic["df"] = Qdf
    
    #scalers = {"Yscaler":Yscaler,"h24scaler":h24scaler}
    #d = {"data":data, "trainX":trainX, "trainY":trainY, "testX":testX, "testY":testY, "dateTrain":dateTrain, "dateTest":dateTest,
    #        "trainPred":ftrp, "trainYtrue":ftry, "testPred":ftep, "testYtrue":ftey,
    #        "modelName":MODEL_NAME, "scalers":scalers}
    return ic


# ## Pruebas

# In[ ]:


STATION, FILTER_YEARS, THETA = get_station("Parque_OHiggins")
qConfig = {
                       "STATION" : STATION,
                        "SCALER" : preprocessing.StandardScaler,
                    "IMPUTATION" : None,
                      "AGREGADOS":[],
                         "TARGET": "O3",
                       "PRECALC" : precalcular_agregados(STATION),
                          "THETA": THETA,
                     "moreTHETA" : [],
                         "FUTURE": 1, #must be 1
                           "PAST": False, #must be False
                        
                  "FILTER_YEARS" : FILTER_YEARS,
                           "CUT" : 0.41,
                "FIXED_FEATURES" : ['CO', 'PM10', 'PM25', 'NO', 'NOX', 'WD', 'RH', 'TEMP', 'WS', 'UVA', 'UVB', 'O3'], # empty list means that the CUT will be used
                           "BAN" : ["countEC", "EC","O3btTHETA"],
                        
                     "FOLDS_TVT" : True,
                      "TIMESTEP" : 7,
                        "OVERLAP": True,
                     
                       "SHUFFLE" : False,
                       "TRAINPCT": 0.85,
    
                "OVERWRITE_MODEL": False,
                    "MODEL_NAME" : "qModel",
                      "QUANTILES": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                        "LAYERS" : [9],
                      "DROP_RATE": [0.376241262719619],
                          "QLOSS" : quantil_loss,
                     "BATCH_SIZE": 16,
                        "EPOCHS" : 400,
                      "PATIENCE" : 20,
                         "GRAPH" : True,
                      "TIMEDIST" : False,  #Sin implementar
    
        }
#Qoutput = quantileLSTM(qConfig)


# In[ ]:


#qConfig["GRAPH"] = False
#temp = Qprediction(Qoutput, qConfig)


# ## Pruebas con varias seeds - preSQP

# In[ ]:


STATION, FILTER_YEARS, THETA = get_station("POH_full")
qConfig = {
                       "STATION" : STATION,
                        "SCALER" : preprocessing.StandardScaler,
                    "IMPUTATION" : None,
                      "AGREGADOS":[],
                         "TARGET": "O3",
                       "PRECALC" : precalcular_agregados(STATION),
                          "THETA": THETA,
                     "moreTHETA" : [],
                         "FUTURE": 1, #must be 1
                           "PAST": False, #must be False
                        
                  "FILTER_YEARS" : FILTER_YEARS,
                           "CUT" : 0.41,
                "FIXED_FEATURES" : ['CO', 'PM10', 'PM25', 'NO', 'NOX', 'WD', 'RH', 'TEMP', 'WS', 'UVA', 'UVB', 'O3'], # empty list means that the CUT will be used
                           "BAN" : ["countEC", "EC","O3btTHETA"],
                        
                     "FOLDS_TVT" : True,
                      "TIMESTEP" : 7,
                        "OVERLAP": True,
                     
                       "SHUFFLE" : False,
                       "TRAINPCT": 0.85,
    
                "OVERWRITE_MODEL": True,
                    "MODEL_NAME" : "qModel",
                      "QUANTILES": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                        "LAYERS" : [9],
                      "DROP_RATE": [0.077202410828456],
                          "QLOSS" : quantil_loss,
                     "BATCH_SIZE": 16,
                        "EPOCHS" : 400,
                      "PATIENCE" : 20,
                         "GRAPH" : False,
                      "TIMEDIST" : False,  #Sin implementar
    
        }

seeds = [123, 57, 872, 340, 77, 583, 101, 178, 938, 555]
all_scoresQ = []
all_outputsQ = []
for s in seeds:
    qConfig["SEED"] = s
    Qoutput = quantileLSTM(qConfig)
    all_outputsQ.append(Qoutput)
    all_scoresQ.append( Qprediction(Qoutput, qConfig) )


# In[ ]:



all_metrics = ["RMSE", "RMSEat%s"%THETA]
for m in all_metrics:
    for d in ['train', 'valid', 'test']:
        fmeans = []
        for i in range( len(seeds) ):
            fmeans.append( np.nanmean(all_scoresQ[i][d][m]) ) #mean over folds
        mean = np.mean(fmeans)
        std  = np.std(fmeans)
        print(m, d, mean, std)


# In[ ]:


STOP


# # Red Selectora

# ## classify
# Obtiene las clases y las marcas, asociadas a cada ejemplo. Recibe una predicción del modelo cuantílico y los Y asociados, ambos en el formato que se obtienen mediante la función Qprediction.

# In[ ]:


def classify(QPRED, QYTRUE, ):
    classY = []
    classMark = []
    for i in range(len(QPRED)):
        # in qpred ----> qn > ... > q2 > q1
        #qpred = QPRED[i, 1:][::-1]
        qpred = QPRED[i, 1:]
        ypred = QPRED[i, 0]
        ytrue = QYTRUE[i, 0]
        newy = []
        newmk = []
        classified = False

        omg = False
        for q1,q2 in zip(qpred[:-1], qpred[1:]):
            #if ypred < q1 and ypred >= q2:
            if ypred > q1 and ypred <= q2:
                newmk.append( ypred )
            else:
                newmk.append( np.mean( (q1,q2) ) )
            
            #if q2 > q1:
            if q1 > q2:
                omg = True
                print("q1 > q2     omg")
                print(qpred, ytrue)
                print(i,q2,q1)
                
            if classified != True:
                #if ytrue < q1 and ytrue >= q2:
                if ytrue > q1 and ytrue <= q2:
                    newy.append(1)
                    classified = True
                else:
                    newy.append(0)
            else:
                newy.append(0)
        
        if classified == False:
            #if ytrue >= qpred[0]:
            if ytrue <= qpred[0]:
                newy[0] = 1
                #print("ERA MAYOR")
            #elif ytrue < qpred[-1]:
            elif ytrue > qpred[-1]:
                newy[-1] = 1
                #print("ERA MENOR")
            else:
                print("NO DEBERIA ESTAR AQUI")
                return
            
            if omg:
                print("OMGGGG")
                #return
                pass
        
        classY.append(newy)
        classMark.append(newmk)
    
    classY = np.array(classY)
    classMark = np.array(classMark)

    return classY, classMark


# ## make_newExamples
# Obtiene las caracteristicas indicadas en subFeatures, recorta los timeStep, y redimensiona el arreglo para que los timestep sean características.

# In[ ]:


def make_newExamples(examples, subFeatures, subTimeStep, d):
    FEATURES = d["data"].columns.tolist()
    subIndex = []
    for s in subFeatures:
        subIndex.append( FEATURES.index(s) )
    
    newExamples = examples[:, -subTimeStep:, subIndex]
    
    a,b,c = newExamples.shape
    newExamples = newExamples.reshape( (a,b*c) )
    
    #if len(examples) > len(d["testY"]):
    #    print(examples.shape)
    #    print(newExamples.shape)
    #    print(d["trainY"].shape)
    #    newExamples = np.hstack([newExamples, d["trainY"][:, -subTimeStep:, 0]])
    #    
    #    
    #else:
    #    newExamples = np.hstack([newExamples, d["testY"][:, -subTimeStep:, 0]])
    
    return newExamples
    


# ## make_classes_and_newExamples
# Crea las clases para las etiquetas de entrenamiento y test.

# In[ ]:


def make_classes_and_newExamples(quantileOutput, selectConfig): #d, subFeatures, subTimeStep):
    #d = quantileOutput
    subFeatures = selectConfig["subFeatures"]
    subTimeStep = selectConfig["subTimeStep"]
    isLSTM = selectConfig["isLSTM"]
    F_LEN = len(subFeatures)
    
    
    print("Making Classes and new Examples ...")
    # Cada fila contiene: (y, q1, q2, ..., qn)
    # donde q1 < q2 < ... < qn
    trainPred = quantileOutput["trainPred"]
    trainYtrue = quantileOutput["trainYtrue"]
    testPred = quantileOutput["testPred"]
    testYtrue = quantileOutput["testYtrue"]
    
    # new Y and classes
    print("trainPred.shape, ", trainPred.shape)
    print("testPred.shape, ", testPred.shape)
    classTrainY, classTrainMark = classify(trainPred, trainYtrue)
    classTestY, classTestMark = classify(testPred, testYtrue)
    print("classTrainMark, ",classTrainMark.shape)
    print("classTrainY.shape, ", classTrainY.shape)
    print("classTestY.shape, ", classTestY.shape)
    
    # new examples
    trainX = quantileOutput["trainX"]
    testX = quantileOutput["testX"]
    
    print("trainX.shape, ",trainX.shape)
    print("testX.shape, ",testX.shape)
    newTrainX = make_newExamples(trainX, subFeatures, subTimeStep,quantileOutput)
    newTestX = make_newExamples(testX, subFeatures, subTimeStep,quantileOutput)
    print("newTrainX.shape, ",newTrainX.shape)
    print("newTestX.shape, ",newTestX.shape)
    
    Yscaler = quantileOutput["scalers"]["Yscaler"]
    
    for tr, te in zip(trainPred[:,0:1].T, testPred[:,0:1].T):
        newTrainX = np.hstack([newTrainX, Yscaler.transform( tr[:, None] ) ])
        newTestX = np.hstack([newTestX, Yscaler.transform( te[: ,None] ) ])
    print("newTrainX.shape, ",newTrainX.shape)
    print("newTestX.shape, ",newTestX.shape)
    
    if isLSTM == True:
        trainX = newTrainX[:,:-1].reshape( (-1, subTimeStep, F_LEN) )
        testX = newTestX[:,:-1].reshape( (-1, subTimeStep, F_LEN) )
        
        #### repeating mean preadiction from quantileLSTM
        ###temp = newTrainX[:,-1]
        ###temp = np.repeat(temp,subTimeStep).reshape(-1, subTimeStep, 1)
        ###trainX = np.concatenate([trainX,temp],axis=2)
        ###temp = newTestX[:,-1]
        ###temp = np.repeat(temp,subTimeStep).reshape(-1, subTimeStep, 1)
        ###testX = np.concatenate([testX,temp],axis=2)
    
        newTrainX = trainX
        newTestX = testX
    

    
    selectD = {"trainY":classTrainY, "trainMarks":classTrainMark, "testY":classTestY, "testMarks":classTestMark,
              "trainX":newTrainX, "testX":newTestX, "quantileOutput":quantileOutput, "F_LEN":F_LEN}
    
    return selectD
    
            
            
        
    

'''
outputForSelect = quantileLSTM(
                        AGREGADOS=[],
                        TARGET= "O3",
                        FUTURE=1, #must be >= 1
                        PAST=False, #must be False or >= 0
                        
                        FEATURES=[],
                        CUT=0.26,
                        BAN=[],
                        
                        TIMESTEP = 5,
                        OVERLAP=True,
                        
                        BATCH_SIZE= 16,
                        TRAINPCT=0.85,
    
                        OVERWRITE_MODEL=False,
                        MODEL_NAME = "qModel",
                        QUANTILES=[0.1, 0.3, 0.5, 0.7, 0.9],
                        LAYERS = [12],
                        EPOCHS = 100,
                        TIMEDIST = False,  #Sin implementar
    
                        )
'''
# ## SQPmodel

# In[ ]:


def SQPmodel(selectConfig, quantileOutput):
    np.random.seed(123)
    set_random_seed(2)
    
    qfilename = quantileOutput["file_name"]
    qLSTM = quantileOutput["model"]
    dfFS = quantileOutput["df"]
    dFS = quantileOutput
    
    subFeatures = selectConfig["subFeatures"]
    subTimeStep = selectConfig["subTimeStep"]
    LAYERS = selectConfig["LAYERS"]
    DROP_RATE = selectConfig["DROP_RATE"]
    EPOCHS = selectConfig["EPOCHS"]
    OVERWRITE_MODEL = selectConfig["OVERWRITE_MODEL"]
    # TIPO DE RED
    isLSTM = selectConfig["isLSTM"]
    
    
    
    #selectD = make_classes_and_newExamples(dFS, subFeatures, subTimeStep)
    #** ic = internal Config **
    ic = make_classes_and_newExamples(quantileOutput, selectConfig)
    
    
    trainX = ic["trainX"]
    trainY = ic["trainY"]
    
    print(np.min(trainY))
    print(np.max(trainY))
    print("trainX.shape,",trainX.shape)
    
    qfileHash = md5(qfilename.encode()).hexdigest()
    ic["qfileHash"] = qfileHash
    FILE_NAME = create_filename(ic, selectConfig)
    
    try:
        print("Model File Name: ",FILE_NAME)
        if OVERWRITE_MODEL == False:
            print("Loading Model...")
            model = load_model(FILE_NAME)
            print(FILE_NAME + " Loaded =)")
        else:
            print("Reentrenando "+ FILE_NAME)
            load_model("noexisto_nijamas_existire.hacheCinco")
    except:
    
        if isLSTM == True:
            #LAYERS=[4]
            #TIMESTEP = subTimeStep
            ###F_LEN = len(subFeatures)
            ###
            ###print(ic["trainX"].shape)
            ###trainX = ic["trainX"][:,:-1].reshape( (-1, subTimeStep, F_LEN) )
            ###temp = ic["trainX"][:,-1]
            ###temp = np.repeat(temp,subTimeStep).reshape(-1, subTimeStep, 1)
            ###print(trainX.shape)
            ###print(temp.shape)
            ###trainX = np.concatenate([trainX,temp],axis=2)
            ###print(trainX.shape)
            ###print(trainY.shape)
            ###
            ###ic["trainX"] = trainX
            
            model = Sequential()
            for Neurons in LAYERS[:-1]:
                model.add(LSTM(Neurons, input_shape=(subTimeStep, trainX.shape[2]), return_sequences=True))
                #model.add(Dropout(rate=DROP_RATE))
            model.add(LSTM(LAYERS[-1], input_shape=(subTimeStep, trainX.shape[2]), return_sequences=False))
            #model.add(Dropout(rate=DROP_RATE))
            model.add(Dense(trainY.shape[1]  , activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            #model.fit( trainX, trainY, epochs=600, batch_size=16  )
            model.fit( trainX, trainY, epochs=EPOCHS, batch_size=16  )
        else:
            #LAYERS=[100,50,50]
            print(trainX.shape)
            print(trainY.shape)
            model = Sequential()
            #model.add(Dense(LAYERS[0], input_dim=trainX.shape[1], activation='relu'))
            #model.add(Dropout(rate=DROP_RATE))
            for Neurons in LAYERS[:-1]:
                model.add(Dense(Neurons, activation='relu'))
                model.add(Dropout(rate=DROP_RATE))
            #model.add(Dense(50, activation='relu'))
            #model.add(Dropout(rate=DROP_RATE))
            model.add(Dense(LAYERS[-1], activation='relu'))
            #model.add(Dropout(rate=DROP_RATE))
            model.add(Dense(trainY.shape[1]  , activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            #model.fit( trainX, trainY, epochs=1000, batch_size=16  )
            model.fit( trainX, trainY, epochs=EPOCHS, batch_size=16  )
            
        model.save(FILE_NAME)
        print(FILE_NAME + " Saved =)")
    
    ic["model"] = model
    return ic


# ## SQPrediction

# In[ ]:


def SQPrediction(internalConfig, selectConfig, quantileOutput):
    np.random.seed(123)
    set_random_seed(2)
    
    isLSTM = selectConfig["isLSTM"]
    subTimeStep = selectConfig["subTimeStep"]
    F_LEN = internalConfig["F_LEN"]
    model = internalConfig["model"]
    trainX = internalConfig["trainX"]
    testX = internalConfig["testX"]
    trainY = internalConfig["trainY"]
    testY = internalConfig["testY"]
    
    
    ###if isLSTM == True:
    ###    print("trainX, ", internalConfig["trainX"].shape)
    ###    print("testX, ", internalConfig["testX"].shape)
    ###    testX = internalConfig["testX"][:,:-1].reshape( (-1, subTimeStep, F_LEN) )
    ###    # adding mean prediciton of quantileLSTM
    ###    temp = internalConfig["testX"][:,-1]
    ###    print("temp, ", temp.shape)
    ###    temp = np.repeat(temp,subTimeStep).reshape(-1, subTimeStep, 1)
    ###    print("testX,", testX.shape)
    ###    print("temp, ", temp.shape)
    ###    testX = np.concatenate([testX,temp],axis=2)
    ###    print(testX.shape)
    ###    
    ###    
    ###    
    ###    #print(ic["trainX"].shape)
    ###    #trainX = ic["trainX"][:,:-1].reshape( (-1, subTimeStep, F_LEN) )
    ###    #temp = ic["trainX"][:,-1]
    ###    #temp = np.repeat(temp,subTimeStep).reshape(-1, subTimeStep, 1)
    ###    #print(trainX.shape)
    ###    #print(temp.shape)
    ###    #trainX = np.concatenate([trainX,temp],axis=2)
    ###    #print(trainX.shape)
    ###    
    ###    
    ###    internalConfig["testX"] = testX
    ###else:
    ###    testX = internalConfig["testX"]
    
    testY = internalConfig["testY"]
    
    caca = model.predict(trainX[0, None])
    print(caca)
    print(trainY[0])
    print(quantileOutput['trainPred'][0])
    a = quantileOutput['trainYq'][0,None]
    print(quantileOutput["scalers"]["Yscaler"].inverse_transform(a))
    
    
    trainPred = model.predict(trainX)
    testPred = model.predict(testX)
    
    ctrainPred = trainPred
    #ctrainPred = []
    #for p in trainPred:
    #    print(type(p.tolist()))
    #    n = [0]*len(p)
    #    n[ np.argmax(p) ] = 1
    #    ctrainPred.append(n)
    print("###########")
    print( np.sum(np.array(ctrainPred),axis=0) )
    print( np.sum(np.array(trainY),axis=0) )
    ctrainPred = np.sum( np.array(ctrainPred) * internalConfig["trainMarks"], axis= 1 )[:, None]
    #print(ctrainPred)
    
    ctestPred = testPred
    #ctestPred = []
    #for p in testPred:
    #    n = [0]*len(p)
    #    n[ np.argmax(p) ] = 1
    #    ctestPred.append(n)
    ctestPred = np.sum( np.array(ctestPred) * internalConfig["testMarks"], axis= 1 )[:, None]
    
    #print("ctestPrede")
    #print(ctestPred)
    
    trainYtrue = quantileOutput["trainYtrue"][:,0, None]
    testYtrue = quantileOutput["testYtrue"][:,0, None]
    
    
    print("### RED SELECTORA ###")
    
    print("Calculando Error")
    trainScore = math.sqrt(mean_squared_error(trainYtrue, ctrainPred))
    print('    Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testYtrue, ctestPred))
    print('    Test Score: %.2f RMSE' % (testScore))
    
    theta=61
    print("Calculando RMSEat%d"%theta)
    trainScore = RMSEat(theta,trainYtrue, ctrainPred)
    print('    Train Score: %.2f RMSEat%d' % (trainScore,theta))
    testScore = RMSEat(theta,testYtrue, ctestPred)
    print('    Test Score: %.2f RMSEat%d' % (testScore, theta))
    
    internalConfig["trainPred"] = ctrainPred
    internalConfig["testPred"] = ctestPred
    
    return internalConfig


# ## SQP

# In[ ]:


def SQP(SQPconfig, quantileConfig):
    for cfg in quantileConfig:
        if cfg not in SQPconfig:
            SQPconfig[cfg] = quantileConfig[cfg]
    quantileOutput = quantileLSTM(quantileConfig)
    internalConfig = SQPmodel(SQPconfig, quantileOutput)
    SQPoutput = SQPrediction(internalConfig, SQPconfig, quantileOutput)
    
    return SQPoutput


# ## Pruebas
# Red cuantílica que alimentará a la selectora, es independite de la cuantílica de pruebas.
'''
# Red cuantílica que alimentará a la selectora, es independite de la cuantílica de pruebas.
qConfigForSelect = {    
                        "AGREGADOS":[],
                        "TARGET": "O3",
                        "PRECALC": precalcular_agregados(),
                        "THETA":61,
                        "FUTURE":1, #must be >= 1
                        "PAST":False, #must be False or >= 0
                        
                        "FIXED_FEATURES":[],
                        "CUT":0.26,
                        "BAN":["countEC","EC"],
                        
                        "TIMESTEP" : 5,
                        "OVERLAP":True,
                        
                        "BATCH_SIZE": 16,
                        "TRAINPCT":0.85,
    
                        "OVERWRITE_MODEL":False,
                        "MODEL_NAME" : "qModel",
                        "QUANTILES":[0.1, 0.3, 0.5, 0.7, 0.9],
                        "LAYERS" : [12],
                        "DROP_RATE": 0.4,
                        "EPOCHS" : 100,
                        "TIMEDIST" : False,  #Sin implementar
                    }

outputForSelect = quantileLSTM(qConfigForSelect)selectConfig = {
        "subFeatures": ['CO', 'PM10', 'PM25', 'NO', 'NOX', 'WD', 'RH', 'TEMP', 'WS', 'UVA', 'UVB','O3'], #d["data"].columns.tolist()[:-6]#
        "subTimeStep": 5,  #debe ser <= con el cual fue entrenada la red cuantilica
        "isLSTM": False,
        "EPOCHS":500,
        "DROP_RATE":0.4,
        "LAYERS":[4],
        "OVERWRITE_MODEL": False,
        "MODEL_NAME": "SQPmodel",
    }

outputSQP = SQPmodel(selectConfig, outputForSelect)
# SQPoutput = SQPrediction(outputSQP, selectConfig, outputForSelect)
### RED SELECTORA ### FFN
Calculando Error
    Train Score: 16.18 RMSE
    Test Score: 14.91 RMSE
Calculando RMSEat61
    Train Score: 14.32 RMSEat61
    Test Score: 12.86 RMSEat61
    
    model = Sequential()
    model.add(Dense(50, input_dim=inputdim, activation='relu'))
    model.add(Dropout(rate=0.4))
    model.add(Dense(25, activation='relu'))
    #model.add(Dropout(rate=0.4))
    #model.add(Dense(25, activation='relu'))
    model.add(Dense(trainY.shape[1]  , activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit( trainX, trainY, epochs=500, batch_size=16  )### RED SELECTORA ### FFN 3Layers
Calculando Error
    Train Score: 17.03 RMSE
    Test Score: 15.42 RMSE
Calculando RMSEat61
    Train Score: 14.75 RMSEat61
    Test Score: 13.82 RMSEat61
    
    model = Sequential()
    model.add(Dense(100, input_dim=inputdim, activation='relu'))
    model.add(Dropout(rate=0.4))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(rate=0.4))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(rate=0.4))
    #model.add(Dense(25, activation='relu'))
    model.add(Dense(trainY.shape[1]  , activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit( trainX, trainY, epochs=500, batch_size=16  )### RED SELECTORA ### LSTM
Calculando Error
    Train Score: 18.45 RMSE
    Test Score: 15.22 RMSE
Calculando RMSEat61
    Train Score: 15.83 RMSEat61
    Test Score: 12.47 RMSEat61
    
    LAYERS=[50, 25]    # tambien con LAYERS=[3]
    model = Sequential()
    for Neurons in LAYERS[:-1]:
        model.add(LSTM(Neurons, input_shape=(TIMESTEP, trainX.shape[2]), return_sequences=True))
        model.add(Dropout(rate=0.4))
    model.add(LSTM(LAYERS[-1], input_shape=(TIMESTEP, trainX.shape[2]), return_sequences=False))
    model.add(Dropout(rate=0.4))
    model.add(Dense(trainY.shape[1]  , activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit( trainX, trainY, epochs=150, batch_size=16  )
'''
# In[ ]:


# Red cuantílica que alimentará a la selectora, es independite de la cuantílica de pruebas.
STATION = "Las_Condes"
quantileConfig = {    
                        "SCALER" : preprocessing.StandardScaler,
                     "AGREGADOS":[],
                        "TARGET": "O3",
                       "PRECALC": precalcular_agregados(STATION),
                         "THETA":61,
                        "FUTURE":1, #must be >= 1
                          "PAST":False, #must be False or >= 0
                        
                "FILTER_YEARS" : [2004,2014],
                         "CUT" : 0.41,
              "FIXED_FEATURES" : ['CO', 'PM10', 'PM25', 'NO', 'NOX', 'WD', 'RH', 'TEMP', 'WS', 'UVA', 'UVB', 'O3'], # empty list means that the CUT will be used
                         "BAN" : ["countEC", "EC","O3btTHETA"],
                        
                     "TIMESTEP" : 28,
                       "OVERLAP":True,
                    
                      "SHUFFLE" : True,
                    "BATCH_SIZE": 16,
                      "TRAINPCT":0.85,
    
               "OVERWRITE_MODEL":False,
                   "MODEL_NAME" : "qModel",
                     "QUANTILES":[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                       "LAYERS" : [25,32],                 
                     "DROP_RATE": [0.0151298677510781, 0.0485420356560454],
                       
                       "EPOCHS" : 400,
                     "PATIENCE" : 20,
                     "TIMEDIST" : False,  #Sin implementar
                    }

SQPconfig = {
        "subFeatures": ['CO', 'PM10', 'PM25', 'NO', 'NOX', 'WD', 'RH', 'TEMP', 'WS', 'UVA', 'UVB','O3'], #d["data"].columns.tolist()[:-6]#
        "subTimeStep": 14,  #debe ser <= con el cual fue entrenada la red cuantilica
        "isLSTM": False,
        "EPOCHS":150,
        "DROP_RATE":0.4,
        "LAYERS":[4],
    
        "OVERWRITE_MODEL": False,
        "MODEL_NAME": "SQPmodel",
    }

#SQPoutput = SQP(SQPconfig,quantileConfig)


# In[ ]:


#dFS = SQPoutput["quantileOutput"]
#dfFS = SQPoutput["quantileOutput"]["df"]
#ctrainPred = SQPoutput["trainPred"]
#ctestPred = SQPoutput["testPred"]

#selectiveX = join_index(dFS["dateTrain"], ctrainPred, "trainSelect")
#selectiveY = join_index(dFS["dateTest"], ctestPred, "testSelect")
#tempSelect = join_index( np.vstack([dFS["dateTrain"][:, None],dFS["dateTest"][:, None]]),
#                         np.vstack([ctrainPred,ctestPred]), "X+Y" )
#tempSelect2 = join_index( dFS["dateTest"][:, None], ctestPred, "OnlyTest")
#toPlot = pd.concat([dfFS[["y","f"]], selectiveX, selectiveY, tempSelect, tempSelect2], axis=1)
#toPlot[["y","f","trainSelect", "testSelect"] ].iplot(title="Comparaciones")


# In[ ]:


#temp = pd.DataFrame(np.abs(toPlot["y"] - toPlot["f"]))
#temp.columns = ["diff f %.3f"%temp[[0]].mean()[0] ]
#
#temp2 = pd.DataFrame(np.abs(toPlot["y"] - toPlot["OnlyTest"] ))
##print(temp2)
#temp2.columns = ["diff selective %.3f "%temp2[[0]].mean()[0] ]
##print(temp2)
#
##print(dfFS["y"])
##print(tempSelect)
#
#diffPlot = pd.concat([temp, temp2],axis=1)
#temp = pd.DataFrame(diffPlot)
#diffPlot = diffPlot.dropna()
#
##diffPlot.iplot(title="Diferencias hacia el Y real")
#
#a = diffPlot.min(axis=1)
#b = diffPlot[ diffPlot.columns[0] ]
#b[b != a] = 0
#b[b == a] = -10
#diffPlot[ diffPlot.columns[0] ] = b
#
#b = diffPlot[ diffPlot.columns[1] ]
#b[b != a] = 0
#b[b == a] = 10
#diffPlot[ diffPlot.columns[1] ] = b
#
#diffPlot.iplot(kind="bar", mode="stack-bar")
#print("Cantidad de veces que predice mejor una sobre la otra")
#print("f, ", abs(diffPlot.sum()/10)[0])
#print("red selectora,",abs(diffPlot.sum()/10)[1])


# In[ ]:


#t = temp.values
#np.sum(t[:,0] == t[:,1])


# # Hyperas

# In[ ]:


from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, randint, quniform
import pickle 


# ## data

# In[ ]:


def data():
    #STATION, FILTER_YEARS, THETA = "Las_Condes", [2004, 2013], 89
    #STATION, FILTER_YEARS, THETA = "Independencia", [2009, 2017], 56
    #STATION, FILTER_YEARS, THETA = "Parque_OHiggins", [2009, 2017], 56
    
    #STATION, FILTER_YEARS, THETA = "Las_Condes", [2008, 2013], 84#[2004, 2013], 89
    #STATION, FILTER_YEARS, THETA = "Independencia", [2009, 2014], 58#[2009, 2017], 56
    #STATION, FILTER_YEARS, THETA = "Parque_OHiggins", [2009, 2014], 60
    STATION, FILTER_YEARS, THETA = get_station("POH_full")
    Config = {
                    # import_merge_and_scale()
                   "STATION" : STATION,
                    "SCALER" : preprocessing.StandardScaler,
                 "AGREGADOS" : [],#["O3","TEMP","WS","RH"],    #["ALL"] #Horas de los maximos que se quieren agregar.
                "IMPUTATION" : None,
                    "TARGET" : "O3",
                   "PRECALC" : [],#precalcular_agregados(),
                     "THETA" : THETA,
                 "moreTHETA" : [],
                    "FUTURE" : 1,
                      "PAST" : False,
              "FILTER_YEARS" : FILTER_YEARS,
                    
                    # select_features()
                       "CUT" : 0.41, #relacionado con FILTER_YEARS
            "FIXED_FEATURES" : ['CO', 'PM10', 'PM25', 'NO', 'NOX', 'WD', 'RH', 'TEMP', 'WS', 'UVA', 'UVB', 'O3'], # empty list means that the CUT will be used
                       "BAN" : ["countEC", "EC","O3btTHETA"], # []
                    
                    #make_examples()
                 "FOLDS_TVT" : True,
                   "OVERLAP" : True,
                     
                     "GRAPH" : False,
                        "Yx" : 0    # DEFAULT 0
                }
    
    np.random.seed(123)
    Config['SHIFT'] = Config['FUTURE'] * -1
    
    scalers = {}
    ic = {"scalers": scalers}
    
    complete_dataset, ylabels, Yscaler, h24scaler = import_merge_and_scale(Config, verbose=False)
    ic["complete_dataset"] = complete_dataset
    ic["ylabels"] = ylabels
    scalers['Yscaler'] = Yscaler
    scalers["h24scaler"] = h24scaler
    
    y_len = len(ic["ylabels"])
    ic["y_len"] = y_len
    
    data, features = select_features(ic, Config)
    ic["data"] = data
    ic["features"] = features
    
    ic["f_len"] = len(ic["features"])
    
    ic["secuencias"] = obtener_secuencias(ic)
    
    
    dtrainX = {}
    dtrainY = {}
    dvalidX = {}
    dvalidY = {}
    dtestX = {}
    dtestY = {}
    ddateTrain = {}
    ddateValid = {}
    ddateTest = {} 
    for LAGS in [1, 2, 3, 5, 7, 14, 21, 28]:
    #for LAGS in [1, 2, 3, 4, 5]:
        
        Config["TIMESTEP"] = LAGS
        examples, y_examples, dateExamples = make_examples(ic, Config, verbose=False)
        ic["examples"] = examples
        ic["y_examples"] = y_examples
        ic["dateExamples"] = dateExamples
        
        d = make_folds_TVT(ic, Config)
        dtrainX[LAGS] = d["list_trainX"]
        dtrainY[LAGS] = d["list_trainY"]
        dvalidX[LAGS] = d["list_validX"]
        dvalidY[LAGS] = d["list_validY"]
        dtestX[LAGS]  = d["list_testX"]
        dtestY[LAGS]  = d["list_testY"]
        ddateTrain[LAGS] = d["list_dateTrain"]
        ddateValid[LAGS] = d["list_dateValid"]
        ddateTest[LAGS]  = d["list_dateTest"]
            
    all_data = {
                "trainX":dtrainX,
                "trainY":dtrainY,
                "validX":dvalidX,
                "validY":dvalidY,
                "testX" :dtestX,
                "testY" :dtestY,
             "dateTrain": ddateTrain,
             "dateValid": ddateValid,
              "dateTest": ddateTest,
                }
            
    filename = 'dictDATA%s.pkl'%(Config["IMPUTATION"])
    output = open(filename, 'wb')
    pickle.dump(all_data, output)
    output.close()
    
    return all_data, scalers, Config


# In[ ]:


a = data()


# In[ ]:


#STOP


# In[ ]:


def HyperData():
    pkl_file = open('dictDATANone.pkl', 'rb')
    all_data = pickle.load(pkl_file)
    pkl_file.close()
    dtrainX=all_data["trainX"]
    dtrainY=all_data["trainY"]
    dvalidX=all_data["validX"]
    dvalidY=all_data["validY"]
    
    return dtrainX, dtrainY, dvalidX, dvalidY


# In[ ]:


#HyperData()


# ## models

# ### HyperLSTM

# In[ ]:


def get_HyperLSTM_Config():
    Config = {
                    #myLSTM()
                "BATCH_SIZE" : 16,
                    "EPOCHS" : 400,
                  "PATIENCE" : 20,
                }
    return Config


# In[ ]:


def HyperLSTM(dtrainX, dtrainY, dvalidX, dvalidY):
    Config = {
                    # import_merge_and_scale()
                "BATCH_SIZE" : 16,
                    
                    #myLSTM()
                    "EPOCHS" : 400,
                  "PATIENCE" : 20,
                }
    
    
    Config = get_HyperLSTM_Config()
        
    #print(f_len)
    BATCH_SIZE = Config["BATCH_SIZE"]
    PATIENCE = Config["PATIENCE"]
    EPOCHS = Config["EPOCHS"]
    
    LAGS = {{choice([1, 2, 3, 5, 7, 14, 21, 28])}}
    #LAGS = {#{choice([1, 2, 3, 4, 5])}}
    TIMESTEP = LAGS
    print("LAGS,", LAGS)
    
    list_trainX = dtrainX[LAGS]
    list_trainY = dtrainY[LAGS]
    list_validX = dvalidX[LAGS]
    list_validY = dvalidY[LAGS]
    
    #print("SPACE")
    #for k in space:
    #    print(k,space[k])
    
    
    
    losses = []
    list_models = []
    for fold in range(0,len(list_trainX)):
        print("Using Fold %s/%s:"%(fold,len(list_trainX)-1))
        trainX = list_trainX[fold]
        trainY = list_trainY[fold]
        validX = list_validX[fold]
        validY = list_validY[fold]
        
    
        f_len = trainX.shape[-1]
        print("trainX.shape,",trainX.shape)
        print("trainY.shape,",trainY.shape)
        print("validX.shape,",validX.shape)
        print("validY.shape,",validY.shape)
        
    
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=PATIENCE, restore_best_weights=True)
        model = Sequential()
        layers={{choice(["one", "two"])}}
        print(layers)
        if  layers == 'two':
            model.add(LSTM(round({{uniform(0,1)}}*10000)%(LAGS*2)+1, activation="sigmoid", input_shape=(TIMESTEP, f_len), return_sequences=True))
            model.add(Dropout(rate={{uniform(0, 1)}}))
        
        model.add(LSTM(round({{uniform(0,1)}}*10000)%(LAGS*2)+1, activation="sigmoid", input_shape=(TIMESTEP, f_len), return_sequences=False))
        model.add(Dropout(rate={{uniform(0, 1)}}))
        model.add(Dense( 1, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer='adam' )
        model.fit( trainX, trainY, epochs=EPOCHS, validation_data=(validX, validY), batch_size=BATCH_SIZE, callbacks=[es], verbose=2)
        loss = model.evaluate(validX, validY, verbose=1)
        print("fold: ",fold, "    loss:",loss)
        losses.append(loss)
        list_models.append(model)
        print("Hora de termino: ",str(datetime.now()))
    
    meanloss = sum(losses)/len(losses)
    print('Valid Loss:', meanloss)
    return {'loss': meanloss, 'status': STATUS_OK, 'model': list_models}


# ### HyperQuantile

# In[ ]:


def get_HyperQuantile_Config():
    Config = {
                    #myLSTM()
                "BATCH_SIZE" : 16,
                    "EPOCHS" : 400,
                  "PATIENCE" : 20,
                }
    return Config


# In[ ]:


def HyperQuantile(dtrainX, dtrainY, dvalidX, dvalidY):
    Config = get_HyperQuantile_Config()
    BATCH_SIZE = Config["BATCH_SIZE"]
    EPOCHS = Config["EPOCHS"]
    PATIENCE = Config["PATIENCE"]
    
    LAGS = {{choice([1, 2, 3, 5, 7, 14, 21, 28])}}
    #LAGS = {#{choice([1, 2, 3, 4, 5])}}
    TIMESTEP = LAGS
    print("LAGS,", LAGS)
    
    list_trainX = dtrainX[LAGS]
    list_trainY = dtrainY[LAGS]
    list_validX = dvalidX[LAGS]
    list_validY = dvalidY[LAGS]
    
    #print("SPACE")
    #for k in space:
    #    print(k,space[k])
    QUANTILES = {{choice([
        [0.1, 0.9],
        [0.3, 0.7],
        [0.1, 0.3, 0.7, 0.9],
        [0.2, 0.4, 0.6, 0.8],
    ])}}
    qlen = len(QUANTILES)
    
    losses = []
    list_models = []
    for fold in range(0,len(list_trainX)):
        print("Using Fold %s-%s:"%(fold,len(list_trainX)-1))
        trainX = list_trainX[fold]
        trainY = list_trainY[fold]
        validX = list_validX[fold]
        validY = list_validY[fold]
        
    
        f_len = trainX.shape[-1]
        print("trainX.shape,",trainX.shape)
        print("trainY.shape,",trainY.shape)
        print("validX.shape,",validX.shape)
        print("validY.shape,",validY.shape)
        
    
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=PATIENCE, restore_best_weights=True)
        qModel = Sequential()
        layers={{choice(["one", "two"])}}
        print(layers)
        if  layers == 'two':
            qModel.add(LSTM(round({{uniform(0,1)}}*10000)%(LAGS*2)+1, activation="sigmoid", input_shape=(TIMESTEP, f_len), return_sequences=True))
            qModel.add(Dropout(rate={{uniform(0, 1)}}))
        
        qModel.add(LSTM(round({{uniform(0,1)}}*10000)%(LAGS*2)+1, activation="sigmoid", input_shape=(TIMESTEP, f_len), return_sequences=False))
        qModel.add(Dropout(rate={{uniform(0, 1)}}))
        qModel.add(Dense( 1 + len(QUANTILES), activation="linear" ))
        
        qModel.compile(loss=lambda y,f: meanquantil_loss2(QUANTILES, 1,qlen,y,f), optimizer='adam')
        
        qModel.fit(trainX, trainY, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(validX,validY), callbacks=[es], verbose=2)
        
        loss = qModel.evaluate(validX, validY, verbose=2)
        print("fold: ",fold, "    loss:",loss)
        losses.append(loss)
        list_models.append(qModel)
        print("Hora de termino: ",str(datetime.now()))
    
    meanloss = sum(losses)/len(losses)
    print('Valid Loss:', meanloss)
    return {'loss': meanloss, 'status': STATUS_OK, 'model': list_models}


# ### HyperPreSQP

# In[ ]:


def get_preSQP_Config():
    Config = {
                    #myLSTM()
                "BATCH_SIZE" : 16,
                    "EPOCHS" : 400,
                  "PATIENCE" : 20,
                 "QUANTILES" : [ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7 ,0.8 ,0.9],
                 #"QUANTILES" : [ 0.75],
                }
    return Config


# In[ ]:


def HyperPreSQP(dtrainX, dtrainY, dvalidX, dvalidY):
    Config = get_preSQP_Config()
    BATCH_SIZE = Config["BATCH_SIZE"]
    EPOCHS = Config["EPOCHS"]
    PATIENCE = Config["PATIENCE"]
    QUANTILES = Config["QUANTILES"]
    
    LAGS = {{choice([1, 2, 3, 5, 7, 14, 21, 28])}}
    #LAGS = {#{choice([1, 2, 3, 4, 5])}}
    TIMESTEP = LAGS
    print("LAGS,", LAGS)
    
    list_trainX = dtrainX[LAGS]
    list_trainY = dtrainY[LAGS]
    list_validX = dvalidX[LAGS]
    list_validY = dvalidY[LAGS]
    
    #print("SPACE")
    #for k in space:
    #    print(k,space[k])

    qlen = len(QUANTILES)
    
    losses = []
    list_models = []
    for fold in range(0,len(list_trainX)):
        print("Using Fold %s-%s:"%(fold,len(list_trainX)-1))
        trainX = list_trainX[fold]
        trainY = list_trainY[fold]
        validX = list_validX[fold]
        validY = list_validY[fold]
        
    
        f_len = trainX.shape[-1]
        print("trainX.shape,",trainX.shape)
        print("trainY.shape,",trainY.shape)
        print("validX.shape,",validX.shape)
        print("validY.shape,",validY.shape)
        
    
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=PATIENCE, restore_best_weights=True)
        qModel = Sequential()
        layers={{choice(["one", "two"])}}
        print(layers)
        if  layers == 'two':
            qModel.add(LSTM(round({{uniform(0,1)}}*10000)%(LAGS*2)+1, activation="sigmoid", input_shape=(TIMESTEP, f_len), return_sequences=True))
            qModel.add(Dropout(rate={{uniform(0, 1)}}))
        
        qModel.add(LSTM(round({{uniform(0,1)}}*10000)%(LAGS*2)+1, activation="sigmoid", input_shape=(TIMESTEP, f_len), return_sequences=False))
        qModel.add(Dropout(rate={{uniform(0, 1)}}))
        qModel.add(Dense( 1 + len(QUANTILES), activation="linear" ))
        
        qModel.compile(loss=lambda y,f: quantil_loss(QUANTILES, 1,qlen,y,f), optimizer='adam')
        
        qModel.fit(trainX, trainY, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(validX,validY), callbacks=[es], verbose=2)
        
        loss = qModel.evaluate(validX, validY, verbose=2)
        print("fold: ",fold, "    loss:",loss)
        losses.append(loss)
        list_models.append(qModel)
        print("Hora de termino: ",str(datetime.now()))
    
    meanloss = sum(losses)/len(losses)
    print('Valid Loss:', meanloss)
    return {'loss': meanloss, 'status': STATUS_OK, 'model': list_models}


# ## runs

# In[ ]:


if "-f" not in sys.argv:
    MODEL_NAME = sys.argv[1]
    MAX_EVALS = int(sys.argv[2])
    print("MODEL_NAME:", MODEL_NAME)
    print("MAX_EVALS:", MAX_EVALS)
else:
    MODEL_NAME = "HyperLSTM"
    MAX_EVALS = 1


if MODEL_NAME == "HyperLSTM":
    HyperModel = HyperLSTM
    get_Config = get_HyperLSTM_Config
    FUNCTIONS = [get_HyperLSTM_Config]

elif MODEL_NAME == "HyperQ":
    HyperModel = HyperQuantile
    get_Config = get_HyperQuantile_Config
    FUNCTIONS = [get_HyperQuantile_Config, meanquantil_loss2]

elif MODEL_NAME == "preSQP":
    HyperModel = HyperPreSQP
    get_Config = get_preSQP_Config
    FUNCTIONS = [get_preSQP_Config, quantil_loss]


# In[ ]:


#STOP


# In[ ]:


best_run, best_model = optim.minimize(model= HyperModel,
                                          data= HyperData,
                                          algo=tpe.suggest,
                                          functions = FUNCTIONS,
                                          max_evals=MAX_EVALS,
                                          trials=Trials(),
                                          notebook_name= "Tisis" if "-f" in sys.argv else None
                                          )


# In[ ]:


print(best_run)


# In[ ]:


if MODEL_NAME == "HyperLSTM":
    LAGS = [1, 2, 3, 5, 7, 14, 21, 28][best_run['LAGS']]
    #LAGS = [1, 2, 3, 4, 5][best_run['LAGS']]
    layers = ['one', 'two'][best_run['layers']]
    lstm1 = round(best_run['round']*10000)%(LAGS*2)+1
    dropout1 = best_run['rate']
    lstm2 = round(best_run['round_1']*10000)%(LAGS*2)+1
    dropout2 = best_run['rate_1']
    
    print("Configuracion:")
    print("        LAGS:",LAGS)
    print("      layers:",layers)
    print("       lstm1:",lstm1)
    print("    dropout1:",dropout1)
    print("       lstm2:",lstm2)
    print("    dropout2:",dropout2)
    print("")
    
    Config = a[2] #Config de data()
    scalers = a[1]
    modelConfig = get_Config()
    Config["TIMEDIST"] = LAGS
    Config["BATCH_SIZE"] = modelConfig["BATCH_SIZE"]
    ic = {}
    ic["scalers"] = scalers
    ic["list_models"] = best_model
    for k in a[0]:
        ic["list_"+k] = a[0][k][LAGS]
    #ic["list_trainX"] = a[0]["trainX"][LAGS]
    #ic["list_trainY"] = a[0]["trainY"][LAGS]
    #ic["list_validX"] = a[0]["validX"][LAGS]
    #ic["list_validY"] = a[0]["validY"][LAGS]
    #ic["list_testX"]  = a[0]["testX"][LAGS]
    #ic["list_testY"]  = a[0]["testY"][LAGS]
    #ic["list_dateTrain"] = a[0]["dateTrain"][LAGS]
    #ic["list_dateValid"] = a[0]["dateValid"][LAGS]
    #ic["list_dateTest"]  = a[0]["dateTest"][LAGS]
    myLSTMPredict(ic,Config)


# In[ ]:


if MODEL_NAME == "HyperQ":
    LAGS = [1, 2, 3, 5, 7, 14, 21, 28][best_run['LAGS']]
    #LAGS = [1, 2, 3, 4, 5][best_run['LAGS']]
    layers = ['one', 'two'][best_run['layers']]
    lstm1 = round(best_run['round']*10000)%(LAGS*2)+1
    dropout1 = best_run['rate']
    lstm2 = round(best_run['round_1']*10000)%(LAGS*2)+1
    dropout2 = best_run['rate_1']
    QUANTILES = [
            [0.1, 0.9],
            [0.3, 0.7],
            [0.1, 0.3, 0.7, 0.9],
            [0.2, 0.4, 0.6, 0.8],
        ][best_run['QUANTILES']]
    
    print("Configuracion:")
    print("         LAGS:",LAGS)
    print("    QUANTILES:",QUANTILES)
    print("       layers:",layers)
    print("        lstm1:",lstm1)
    print("     dropout1:",dropout1)
    print("        lstm2:",lstm2)
    print("     dropout2:",dropout2)
    print("")
    
    Config = a[2] #Config de data()
    scalers = a[1]
    modelConfig = get_Config()
    Config["TIMEDIST"] = LAGS
    Config["QUANTILES"] = QUANTILES
    Config["BATCH_SIZE"] = modelConfig["BATCH_SIZE"]
    ic = {}
    ic["scalers"] = scalers
    ic["list_models"] = best_model
    for k in a[0]:
        ic["list_"+k] = a[0][k][LAGS]
    #ic["list_trainX"] = a[0]["trainX"][LAGS]
    #ic["list_trainY"] = a[0]["trainY"][LAGS]
    #ic["list_validX"] = a[0]["validX"][LAGS]
    #ic["list_validY"] = a[0]["validY"][LAGS]
    #ic["list_testX"]  = a[0]["testX"][LAGS]
    #ic["list_testY"]  = a[0]["testY"][LAGS]
    #ic["list_dateTrain"] = a[0]["dateTrain"][LAGS]
    #ic["list_dateValid"] = a[0]["dateValid"][LAGS]
    #ic["list_dateTest"]  = a[0]["dateTest"][LAGS]
    
    
    Qprediction(ic, Config)
    print("mean_loss2, meanerror + meanLq")
    print('''QUANTILES = {{choice([
            [0.1, 0.9],
            [0.3, 0.7],
            [0.1, 0.3, 0.7, 0.9],
            [0.2, 0.4, 0.6, 0.8],
        ])}}''')


# In[ ]:


if MODEL_NAME == "preSQP":
    LAGS = [1, 2, 3, 5, 7, 14, 21, 28][best_run['LAGS']]
    #LAGS = [1, 2, 3, 4, 5][best_run['LAGS']]
    layers = ['one', 'two'][best_run['layers']]
    lstm1 = round(best_run['round']*10000)%(LAGS*2)+1
    dropout1 = best_run['rate']
    lstm2 = round(best_run['round_1']*10000)%(LAGS*2)+1
    dropout2 = best_run['rate_1']
    modelConfig = get_Config()
    QUANTILES = modelConfig["QUANTILES"]
    
    print("Configuracion:")
    print("         LAGS:",LAGS)
    print("    QUANTILES:",QUANTILES)
    print("       layers:",layers)
    print("        lstm1:",lstm1)
    print("     dropout1:",dropout1)
    print("        lstm2:",lstm2)
    print("     dropout2:",dropout2)
    print("")
    
    Config = a[2] #Config de data()
    scalers = a[1]
    
    Config["TIMEDIST"] = LAGS
    Config["QUANTILES"] = QUANTILES
    Config["BATCH_SIZE"] = modelConfig["BATCH_SIZE"]
    ic = {}
    ic["scalers"] = scalers
    ic["list_models"] = best_model
    for k in a[0]:
        ic["list_"+k] = a[0][k][LAGS]
    
    Qprediction(ic, Config)
    print("mean_loss2, meanerror + meanLq")


# In[ ]:




