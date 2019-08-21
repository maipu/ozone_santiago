#!/usr/bin/env python
# coding: utf-8




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

#asdf = precalcular_eventos(61,"Independencia",REMAKE=False)


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
STATION, FILTER_YEARS, THETA = get_station("Las_Condes")
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
        "THETA":89,
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
        "THETA":89,
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
    STATION, FILTER_YEARS, THETA = get_station("Las_Condes")
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




# ## Pruebas con varias seeds

# In[ ]:





# In[ ]:


STATION, FILTER_YEARS, THETA = get_station("Las_Condes")
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
                  "TIMESTEP" : 28,
                   "OVERLAP" : True,
                    
                    #make_traintest() Ignored when FOLDS_TVT == True
                   "SHUFFLE" : False,
                  "TRAINPCT" : 0.85,
                   
                    
                    #myLSTM()
           "OVERWRITE_MODEL" : False,
                "MODEL_NAME" : "classicModel",
                    "LAYERS" : [18, 51],
                  "DROP_RATE": [0.0970659473593013, 0.163032727137165],
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
'''
seeds = [123, 57, 872, 340, 77, 583, 101, 178, 938, 555]
all_scores = []
all_outputs = []
for s in seeds:
    LSTMconfig["SEED"] = s
    myLSTMoutput = myLSTM(LSTMconfig)
    all_outputs.append(myLSTMoutput)
    all_scores.append( myLSTMPredict(myLSTMoutput, LSTMconfig) )
'''


# In[ ]:


'''
all_metrics = ["RMSE", "RMSEat%s"%THETA]
for m in all_metrics:
    for d in ['train', 'valid', 'test']:
        fmeans = []
        for i in range( len(seeds) ):
            fmeans.append( np.nanmean(all_scores[i][d][m]) ) #mean over folds
        mean = np.mean(fmeans)
        std  = np.std(fmeans)
        print(m, d, mean, std)
'''


# In[ ]:


#STOP





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


# ## Pruebas con varias seeds - preSQP

# In[ ]:


STATION, FILTER_YEARS, THETA = get_station("Las_Condes")
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
                      "TIMESTEP" : 28,
                        "OVERLAP": True,
                     
                       "SHUFFLE" : False,
                       "TRAINPCT": 0.85,
    
                "OVERWRITE_MODEL": False,
                    "MODEL_NAME" : "qModel",
                      "QUANTILES": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                        "LAYERS" : [45],
                      "DROP_RATE": [0.103909704884679],
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
import pickle
output = open("CONDES_all_scoresQ.pkl", 'wb')
pickle.dump(all_scoresQ, output)
output.close()

