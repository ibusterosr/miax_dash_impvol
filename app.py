from bs4 import BeautifulSoup
import requests
import pandas as pd
from datetime import datetime
import numpy as np
import mibian
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px
import math
import os


def norm_pdf(x):
    """
    Función de densidad de probabilidad normal.
    """
    return (1.0/(2*math.pi)**0.5)*math.exp(-0.5*x*x)

def norm_cdf(x):
    """
    Función de distribución acumulada normal.
    """
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def d1(S, K, r, sigma, T):
    """
    Cálculo del parámetro d1 para el modelo de Black-Scholes.
    """
    return (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))

def d2(S, K, r, sigma, T):
    """
    Cálculo del parámetro d2 para el modelo de Black-Scholes.
    """
    return d1(S, K, r, sigma, T) - sigma*math.sqrt(T)

def bs_call_price(S, K, r, sigma, T):
    """
    Calcula el precio teórico de una opción de compra (call) utilizando el modelo de Black-Scholes.
    """
    d1 = (math.log(S/K) + (r + sigma**2/2)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    N1 = 0.5*(1 + math.erf(d1/math.sqrt(2)))
    N2 = 0.5*(1 + math.erf(d2/math.sqrt(2)))
    return S*N1 - K*math.exp(-r*T)*N2

def bs_call_implied_vol(S, K, r, T, C, tol=1e-6, max_iter=100):
    """
    Calcula la volatilidad implícita de una opción de compra (call) utilizando el método de Newton-Raphson.
    """
    sigma = 0.5 # Suposición inicial de la volatilidad implícita
    for i in range(max_iter):
        price = bs_call_price(S, K, r, sigma, T)
        vega = S * math.sqrt(T) * norm_pdf(d1(S, K, r, sigma, T))
        diff = price - C
        if abs(diff) < tol:
            return sigma
        if vega == 0:
            sigma = sigma
        else:
            sigma = sigma - diff / vega
    return None



def bs_put_price(S, K, r, sigma, T):
    """
    Cálculo del precio teórico de una opción de venta utilizando el modelo de Black-Scholes.
    """
    d_1 = d1(S, K, r, sigma, T)
    d_2 = d2(S, K, r, sigma, T)
    return K*math.exp(-r*T)*norm_cdf(-d_2) - S*norm_cdf(-d_1)

def bs_put_implied_vol(S, K, r, T, P, tol=1e-6, max_iter=100):
    """
    Calcula la volatilidad implícita de una opción de venta (put) utilizando el método de Newton-Raphson.
    """
    sigma = 0.5 # Suposición inicial de la volatilidad implícita
    for i in range(max_iter):
        price = bs_put_price(S, K, r, sigma, T)
        vega = S * math.sqrt(T) * norm_pdf(d1(S, K, r, sigma, T))
        diff = price - P
        if abs(diff) < tol:
            return sigma
        if vega == 0:
            sigma = sigma
        else:
            sigma = sigma - diff / vega
    return None




url = 'https://www.meff.es/esp/Derivados-Financieros/Ficha/FIEM_MiniIbex_35'
page = requests.get(url)

soup = BeautifulSoup(page.text, 'html.parser')
tabla = soup.find_all("table", id="tblOpciones")

fechas = tabla[0].find_all("option")
l_fech = []
for f in fechas[:-1]:
    #print(f['value'])
    l_fech.append(f['value'])


call_list = []
put_list = []
for elemento in l_fech:
    if elemento.startswith('OCE'):
        call_list.append(elemento)
    elif elemento.startswith('OPE'):
        put_list.append(elemento)


dicc_call = {}
for n in range(len(call_list)):
    #print(n)
    values = tabla[0].find_all("tr", {"data-tipo":f"{call_list[n]}"})
    df_matriz = []

    for row in values:
        row_df = []

        for j in row:
            row_df.append(j.text)
        
        df_matriz.append(row_df)

    matriz = np.array(df_matriz)

    df = pd.DataFrame(matriz)
    df = df.iloc[:,[1,13]]
    nuevos_nombres = {1: 'Strike', 13: 'ANT'}
    df.rename(columns=nuevos_nombres, inplace=True)
    dicc_call[call_list[n]] = df


dicc_put = {}
for n in range(len(put_list)):
    #print(n)
    values = tabla[0].find_all("tr", {"data-tipo":f"{put_list[n]}"})
    df_matriz = []

    for row in values:
        row_df = []

        for j in row:
            row_df.append(j.text)
        
        df_matriz.append(row_df)

    matriz = np.array(df_matriz)

    df = pd.DataFrame(matriz)
    df = df.iloc[:,[1,13]]
    nuevos_nombres = {1: 'Strike', 13: 'ANT'}
    df.rename(columns=nuevos_nombres, inplace=True)
    dicc_put[put_list[n]] = df


tb_futuros = soup.find_all("table", id="Contenido_Contenido_tblFuturos")
row_futuros = tb_futuros[0].find("tr", {"class":"text-right"})
data = []
for i in row_futuros:
    data.append(i.text)

p_suby = float(data[-2].replace('.', '').replace(',', '.'))


def str_to_float(x):
    if x == "- \xa0":
        return 1
    else:
        return float(x.replace('.', '').replace(',', '.'))
    

for key_call in dicc_call.keys():
    fecha_str = key_call

    # Extraer la fecha como string y convertirla a formato fecha
    fecha = datetime.strptime(fecha_str[3:], '%Y%m%d').date()
    hoy = datetime.now().date()
    diferencia = (fecha - hoy).days

    dicc_call[key_call]['Strike'] = dicc_call[key_call]['Strike'].apply(str_to_float)
    dicc_call[key_call]['ANT'] = dicc_call[key_call]['ANT'].apply(str_to_float)
    dicc_call[key_call]['Uprice'] = p_suby * np.ones(dicc_call[key_call].shape[0])
    dicc_call[key_call]['Irate'] = np.zeros(dicc_call[key_call].shape[0])
    dicc_call[key_call]['DaysExp'] = int(diferencia) * np.ones(dicc_call[key_call].shape[0])
    dicc_call[key_call]['ImpVol'] = np.zeros(dicc_call[key_call].shape[0])



for key_put in dicc_put.keys():
    fecha_str = key_put

    # Extraer la fecha como string y convertirla a formato fecha
    fecha = datetime.strptime(fecha_str[3:], '%Y%m%d').date()
    hoy = datetime.now().date()
    diferencia = (fecha - hoy).days

    dicc_put[key_put]['Strike'] = dicc_put[key_put]['Strike'].apply(str_to_float)
    dicc_put[key_put]['ANT'] = dicc_put[key_put]['ANT'].apply(str_to_float)
    dicc_put[key_put]['Uprice'] = p_suby * np.ones(dicc_put[key_put].shape[0])
    dicc_put[key_put]['Irate'] = np.zeros(dicc_put[key_put].shape[0])
    dicc_put[key_put]['DaysExp'] = int(diferencia) * np.ones(dicc_put[key_put].shape[0])
    dicc_put[key_put]['ImpVol'] = np.zeros(dicc_put[key_put].shape[0])


for call_key in dicc_call.keys():

    df_calc = dicc_call[call_key]
    uprice = df_calc['Uprice']
    strike = df_calc['Strike']
    irate = df_calc['Irate']
    daysexp = df_calc['DaysExp']
    callprice = df_calc['ANT']

    for i in range(df_calc.shape[0]):
        #c = mibian.BS([uprice[i], strike[i], irate[i],daysexp[i]], callPrice=callprice[i])
        iv = bs_call_implied_vol(S=uprice[i], K=strike[i], r=0, T=daysexp[i], C=callprice[i])
        if iv is None:
            iv = 0
        dicc_call[call_key]['ImpVol'][i] = iv



for put_key in dicc_put.keys():

    df_calc = dicc_put[put_key]
    uprice = df_calc['Uprice']
    strike = df_calc['Strike']
    irate = df_calc['Irate']
    daysexp = df_calc['DaysExp']
    putprice = df_calc['ANT']

    for i in range(df_calc.shape[0]):
        #c = mibian.BS([uprice[i], strike[i], irate[i],daysexp[i]], putPrice=putprice[i])
        iv = bs_put_implied_vol(S=uprice[i], K=strike[i], r=0, T=daysexp[i], P=putprice[i])
        if iv is None:
            iv = 0
        dicc_put[put_key]['ImpVol'][i] = iv
        dicc_put[put_key]['ImpVol'][i] = iv


print('OK')
#print(dicc_put)



###################
###################
###################
###################


app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('Comparison between Implied Volatility and Call/Put Option Strike', style={'textAlign': 'center', 'marginTop': '50px', 'marginBottom': '30px', 'backgroundColor': 'darkgreen', 'color': 'white', 'height': '80px'}),
    html.Div([
        dcc.Dropdown(
            id='tipo-opcion',
            options=[{'label': i, 'value': i} for i in ['Call', 'Put']],
            value='Call',
            style={'width': '45%', 'margin-left': '100px'}
        ),
        dcc.Dropdown(
            id='fecha',
            value=call_list[0],
            style={'width': '35%', 'margin-right': '100px'}
        )
    ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'marginTop': '30px'}),
    html.Div([
        html.Span('💲', style={'fontSize': '30px', 'margin': '0px 10px'}),
        html.Span('💲', style={'fontSize': '30px', 'margin': '0px 10px'}),
        html.Span('💲', style={'fontSize': '30px', 'margin': '0px 10px'}),
    ], style={'display': 'flex', 'justify-content': 'center', 'marginTop': '30px'}),
    html.Iframe(id='graph-frame', style={'border': '1px solid green', 'width': '100%', 'height': '500px', 'marginTop': '30px'})
])

@app.callback(Output('fecha', 'options'),
              [Input('tipo-opcion', 'value')])
def set_fechas_options(selected_tipo_opcion):
    if selected_tipo_opcion == 'Call':
        return [{'label': i, 'value': i} for i in call_list]
    elif selected_tipo_opcion == 'Put':
        return [{'label': i, 'value': i} for i in put_list]
    

@app.callback(Output('fecha', 'value'),
              [Input('tipo-opcion', 'value')])
def set_fecha_value(selected_tipo_opcion):
    if selected_tipo_opcion == 'Call':
        return call_list[0]
    elif selected_tipo_opcion == 'Put':
        return put_list[0]
    
@app.callback(Output('graph-frame', 'srcDoc'),
              [Input('tipo-opcion', 'value'),
               Input('fecha', 'value')])
def update_figure(option_type, option_date):
    if option_type == 'Call':
        df = dicc_call[option_date]
    else:
        df = dicc_put[option_date]
    fig = px.line(df, x='Strike', y='ImpVol')
    return fig.to_html()
    
if __name__ == '__main__':
    app.run_server(host='0.0.0.0', debug=True, port=os.getenv('PORT', 8080))