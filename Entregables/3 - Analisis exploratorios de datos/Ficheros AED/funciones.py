import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px 
import re
from fuzzywuzzy import process
import warnings
warnings.filterwarnings("ignore")


def check_df(df, tipo=''):
    if tipo == 'simple':
        print("¿Cuántas filas y columnas hay en el conjunto de datos?")
        num_filas, num_columnas = df.shape
        print("\tHay {:,} filas y {:,} columnas.".format(num_filas, num_columnas))

        print("¿Cuáles son las primeras dos filas del conjunto de datos?")
        display(df.head(2))
        print('\n########################################################################################')
    else:
        print("¿Cuántas filas y columnas hay en el conjunto de datos?")
        num_filas, num_columnas = df.shape
        print("\tHay {:,} filas y {:,} columnas.".format(num_filas, num_columnas))
        print('\n########################################################################################')

        print("¿Cuáles son las primeras cinco filas del conjunto de datos?")
        display(df.head())
        print('\n########################################################################################')

        print("¿Cuáles son las últimas cinco filas del conjunto de datos?")
        display(df.tail())
        print('\n########################################################################################')

        print("¿Cómo puedes obtener una muestra aleatoria de filas del conjunto de datos?")
        display(df.sample(n = 5))
        print('\n########################################################################################')

        print("¿Cuáles son las columnas del conjunto de datos? ¿Cuál es el tipo de datos de cada columna?")
        print(df.dtypes)
        print('\n########################################################################################')

        print("¿Cuántas columnas hay de cada tipo de datos?")
        print(df.dtypes.value_counts())
        print('\n########################################################################################')

        print("¿Cuáles son las variables numéricas?")
        df_numericas = df.select_dtypes(include = 'number')
        columnas_numericas = list(df_numericas.columns)
        print(columnas_numericas)
        print('\n########################################################################################')

        print("¿Cuáles son las variables categóricas?")
        df_categoricas = df.select_dtypes(include = 'object')
        columnas_categoricas = list(df_categoricas.columns)
        print(columnas_categoricas)
        print('\n########################################################################################')

        print("¿Cuántos valores únicos tiene cada columna?")
        print(df.nunique())
        print('\n########################################################################################')

        if len(columnas_numericas)>0:
            print("¿Cuáles son las estadísticas descriptivas básicas de las columnas numéricas?")
            display(df.describe(include = 'number'))
            print('\n########################################################################################')

        if len(columnas_categoricas)>0:
            print("¿Cuáles son las estadísticas descriptivas básicas de las columnas categóricas?")
            display(df.describe(include = 'object'))
        
        return df_numericas, df_categoricas

def identificacion_valores_problem(df, columnas=[]):
    print('###################################################################################')
    print('3.1.1. Proporción de NULOS en cada una de las columnas del conjunto de datos:')
    print(round((df.isnull().sum()/len(df))*100, 2).sort_values(ascending= False))
    print('###################################################################################')
    print(f'3.1.2. Número de DUPLICADOS totales: {df.duplicated().sum()}')
    print('###################################################################################')
    if len(columnas) > 0:
        print(f'3.1.2. Número de DUPLICADOS parciales según las columnas {columnas}: {df.duplicated(subset=columnas).sum()}')
        print('###################################################################################')
    df_numericas = df.select_dtypes(include = 'number')
    columnas_numericas = list(df_numericas.columns)
    if len(columnas_numericas) > 0:
        print('3.1.3. Columnas numéricas con OUTLIERS')
        for var in columnas_numericas:
            Q1 = df[var].quantile(0.25)
            Q3 = df[var].quantile(0.75)
            limite_inferior = Q1 - 1.5 * (Q3 - Q1)
            limite_superior = Q3 + 1.5 * (Q3 - Q1)
            outliers = df[(df[var] < limite_inferior) | (df[var] > limite_superior)]
            print(f'Número de outliers en la columna "{var}": {outliers.shape[0]}')
        print('###################################################################################')

def procesar_fecha(fecha):
  '''
    * Separados por "-":
      - Patrón 1: 04-01-2020
      - Patrón 2: 2020-01-10
      - Patrón 3: 01-14-20

    * Separados por "/":
      - Patrón 4: 11/01/2020
      - Patrón 5: 02/03/20
  '''

  # Separador '-'

  # %d-%m-%y'
  patron1 = r'^(0[1-9]|[12][0-9]|3[01])-(0[1-9]|1[0-2])-(\d{2})$'
  # dia: (0[1-9]|[12][0-9]|3[01])
  # mes: (0[1-9]|1[0-2])
  # año: (\d{2})

  #'%d-%m-%Y'
  patron2 = r'^(0[1-9]|[12][0-9]|3[01])-(0[1-9]|1[0-2])-(\d{4})$'

  #'%m-%d-%y'
  patron3 = r'^(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])-(\d{2})$'

  #'%m-%d-%Y'
  patron4 = r'^(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])-(\d{4})$'

  #'%Y-%m-%d'
  patron5 = r'^(\d{4})-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])$'

  # Separador '/'

  #'%d/%m/%y'
  patron6 = r'^(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[0-2])/(\d{2})$'

  #'%m/%d/%y'
  patron7 = r'^(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])/(\d{2})$'

  #'%m/%d/%Y'
  patron8 = r'^(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])/(\d{4})$'

  #'%Y/%m/%d'
  patron9 = r'^(\d{4})/(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])$'

  #'%Y/%m/%d'
  patron10 = r'^(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[0-2])/(\d{4})$'

  # 12/5/2021	
  #'%Y/%m/%d'
  patron11 = r'^(0?[1-9]|1[0-2])/(0?[1-9]|[12][0-9]|3[01])/(\d{4})$'

  # Comprueba si la fecha cumple con el patrón
  if pd.notnull(fecha) and re.fullmatch(patron1, fecha):
    # Parsea la fecha al formato deseado y devuelve en formato "aaaa-mm-dd"
    return pd.to_datetime(fecha, format='%d-%m-%y').strftime('%Y-%m-%d')

  elif pd.notnull(fecha) and re.fullmatch(patron2, fecha):
    return pd.to_datetime(fecha, format='%d-%m-%Y').strftime('%Y-%m-%d')

  elif pd.notnull(fecha) and re.fullmatch(patron3, fecha):
    return pd.to_datetime(fecha, format='%m-%d-%y').strftime('%Y-%m-%d')

  elif pd.notnull(fecha) and re.fullmatch(patron4, fecha):
    return pd.to_datetime(fecha, format='%m-%d-%Y').strftime('%Y-%m-%d')

  elif pd.notnull(fecha) and re.fullmatch(patron5, fecha):
    return pd.to_datetime(fecha, format='%Y-%m-%d').strftime('%Y-%m-%d')

  elif pd.notnull(fecha) and re.fullmatch(patron6, fecha):
    return pd.to_datetime(fecha, format='%d/%m/%y').strftime('%Y-%m-%d')

  elif pd.notnull(fecha) and re.fullmatch(patron7, fecha):
      return pd.to_datetime(fecha, format='%m/%d/%y').strftime('%Y-%m-%d')

  elif pd.notnull(fecha) and re.fullmatch(patron8, fecha):
      return pd.to_datetime(fecha, format='%m/%d/%Y').strftime('%Y-%m-%d')

  elif pd.notnull(fecha) and re.fullmatch(patron9, fecha):
      return pd.to_datetime(fecha, format='%Y/%m/%d').strftime('%Y-%m-%d')

  elif pd.notnull(fecha) and re.fullmatch(patron10, fecha):
      return pd.to_datetime(fecha, format='%d/%m/%Y').strftime('%Y-%m-%d')
  
  # 12/5/2021
  elif pd.notnull(fecha) and re.fullmatch(patron11, fecha):
      return pd.to_datetime(fecha, format='%m/%d/%Y').strftime('%Y-%m-%d')
 
  else:
      # Devuelve la fecha original si no cumple con el patrón o es NaN
      return pd.NaT  # Retorna Not a Time para fechas que no coinciden con ningún formato
  
def deteccion_outliers (df, columna):
    # Calcular Q1, Q3 e IQR
    Q1 = df[columna].quantile(0.25)
    Q3 = df[columna].quantile(0.75)
    IQR = Q3 - Q1

    # Definir límites inferior y superior para detectar outliers
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR

    print(f"Los valores atípicos se definen como aquellos que caen fuera del siguiente rango:")
    print(f"\t - Límite inferior (considerado extremadamente bajo): {limite_inferior:.2f}")
    print(f"\t - Límite superior (considerado extremadamente alto): {limite_superior:.2f}")

    # Identificar los outliers
    outliers = df[(df[columna] < limite_inferior) | (df[columna] > limite_superior)]

    print(f'Número de outliers en la columna "{columna}": {outliers.shape[0]}')
    return outliers

def limpiar_eventos(df, columna):
    originales = df[columna].copy()
    # Extraer lo que esté dentro de paréntesis como 'Fecha'
    df["Fecha"] = originales.apply(lambda x: re.search(r"\((.*?)\)", x).group(1).strip() if re.search(r"\((.*?)\)", x) else "")
    # Sobrescribir 'Eventos' hasta 'kg'
    df[columna] = originales.apply(lambda x: x.split("kg")[0].strip() + "kg")
    return df   

def transformar (df, value_vars, var_name, value_name, id_vars):
   df_melt = df.melt(
    value_vars = value_vars, 
    var_name = var_name, 
    value_name = value_name,
    id_vars = id_vars)
   return df_melt

def unir_fecha_anio(df, col_fecha, col_anio, nueva_fecha):
    # Concatenar columnas de fecha y año
    df[nueva_fecha] = df[col_fecha].astype(str) + "." + df[col_anio].astype(str)
    # Pasar a formato datetime
    df[nueva_fecha] = pd.to_datetime(df[nueva_fecha], format="%d.%m.%Y", errors="coerce")
    # Eliminar columnas y renombrar
    df.drop(columns=[col_fecha,col_anio],inplace=True)
 

    return df

def convertir_tipos(df):
    # Columnas object (str)
    df["Genero"] = df["Genero"].astype(str)
    df["Categoria"] = df["Categoria"].astype(str) 
    df["Atleta"] = df["Atleta"].astype(str)
    df["Pais"] = df["Pais"].astype(str)

    # Columnas datetime
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")

    # Columna categorica
    orden_medallas = ["Bronce", "Plata", "Oro"]
    df["Medalla"] = pd.Categorical(df["Medalla"],
                                   categories=orden_medallas,
                                   ordered=True)

    # Columnas numericas(enteros)
    df["Arrancada"] = pd.to_numeric(df["Arrancada"], errors="coerce").astype("Int64")
    df["Dos Tiempos"] = pd.to_numeric(df["Dos Tiempos"], errors="coerce").astype("Int64")
    df["Total"] = pd.to_numeric(df["Total"], errors="coerce").astype("Int64")

    return df

    df_copia = df.copy()
    for columna in columnas:
        valores_no_nulos = df_copia[columna].dropna().values
        df_copia[columna] = df_copia[columna].apply(lambda x: np.random.choice(valores_no_nulos) if pd.isna(x) else x)
    return df_copia

import requests



import requests

def funcion_request(url):
    try:
        page = requests.get(url)
        if page.status_code == 403:
            raise ValueError("Acceso prohibido, intentando con un User-Agent modificado...\n")
        print(f'Estado de la petición: {page.status_code} - {page.reason}')
        return page
    except ValueError as ve:
        print(ve)
        headers = {'User-Agent': 'Mozilla/5.0'}
        page = requests.get(url, headers=headers)
        print(f'Reintento - Estado de la petición: {page.status_code} - {page.reason}')
        return page
    except Exception as e:
        print(f'Error al realizar la petición: {e}')
        return None

Lista_paises = [
    "Afganistán","Albania","Alemania","Andorra","Angola","Antigua y Barbuda","Arabia Saudita","Argelia","Argentina",
    "Armenia","Australia","Austria","Azerbaiyán","Bahamas","Bangladés","Barbados","Baréin","Bélgica","Belice","Benín",
    "Bielorrusia","Birmania","Bolivia","Bosnia y Herzegovina","Botsuana","Brasil","Brunéi","Bulgaria","Burkina Faso",
    "Burundi","Bután","Cabo Verde","Camboya","Camerún","Canadá","Catar","Chad","Chile","China","Chipre","Ciudad del Vaticano",
    "Colombia","Comoras","Corea del Norte","Corea del Sur","Costa de Marfil","Costa Rica","Croacia","Cuba","Dinamarca",
    "Dominica","Ecuador","Egipto","El Salvador","Emiratos Árabes Unidos","Eritrea","Eslovaquia","Eslovenia","España",
    "Estados Unidos","Estonia","Esuatini","Etiopía","Filipinas","Finlandia","Fiyi","Francia","Gabón","Gambia","Georgia",
    "Ghana","Granada","Grecia","Guatemala","Guyana","Guinea","Guinea Ecuatorial","Guinea-Bisáu","Haití","Honduras",
    "Hungría","India","Indonesia","Irak","Irán","Irlanda","Islandia","Islas Marshall","Islas Salomón","Israel","Italia",
    "Jamaica","Japón","Jordania","Kazajistán","Kenia","Kirguistán","Kiribati","Kuwait","Laos","Lesoto","Letonia","Líbano",
    "Liberia","Libia","Liechtenstein","Lituania","Luxemburgo","Macedonia del Norte","Madagascar","Malasia","Malaui",
    "Maldivas","Malta","Marruecos","Mauricio","Mauritania","México","Micronesia","Moldavia","Mónaco","Mongolia","Montenegro",
    "Mozambique","Namibia","Nauru","Nepal","Nicaragua","Níger","Nigeria","Noruega","Nueva Zelanda","Omán","Países Bajos",
    "Pakistán","Palaos","Panamá","Papúa Nueva Guinea","Paraguay","Perú","Polonia","Portugal","Reino Unido","República Centroafricana",
    "República Checa","República Dominicana","Ruanda","Rumanía","Rusia","Samoa","San Cristóbal y Nieves","San Marino",
    "San Vicente y las Granadinas","Santa Lucía","Santo Tomé y Príncipe","Senegal","Serbia","Seychelles","Sierra Leona",
    "Singapur","Siria","Somalia","Sri Lanka","Sudáfrica","Sudán","Sudán del Sur","Suecia","Suiza","Surinam","Tailandia",
    "Tanzania","Tayikistán","Timor Oriental","Togo","Tonga","Trinidad y Tobago","Túnez","Turkmenistán","Turquía","Tuvalu",
    "Ucrania","Uganda","Uruguay","Uzbekistán","Vanuatu","Venezuela","Vietnam","Yemen","Yibuti","Zambia","Zimbabue"
]


def separar_numeros_atletas(df, columna):
    # Inicializar columnas en el mismo DF
    df["Arrancada"] = None
    df["Dos Tiempos"] = None
    df["Total"] = None

    for i, fila in df[columna].items():
        # Buscar números de 2 o 3 cifras
        numeros = re.findall(r"\b\d{2,3}\b", str(fila))

        if len(numeros) >= 1:
            df.at[i, "Arrancada"] = numeros[0]
        if len(numeros) >= 2:
            df.at[i, "Dos Tiempos"] = numeros[1]
        if len(numeros) >= 3:
            df.at[i, "Total"] = numeros[2]

    return df

def separar_paises(df, columna, threshold=85):
    # Inicializar columna en el mismo DF
    df["Pais"] = None

    for i, fila in df[columna].items():
        tokens = str(fila).split()
        pais_detectado = None

        for tok in tokens:
            match, score = process.extractOne(tok, Lista_paises)
            if score >= threshold:
                pais_detectado = match
                break

        df.at[i, "Pais"] = pais_detectado

    return df

def separar_atleta(df, columna):
    # Crear nueva columna "Atleta" con las 3 primeras palabras
    df["Atleta"] = df[columna].apply(
        lambda x: " ".join(str(x).split()[:3]) if pd.notna(x) else None
    )
    return df


