import dash
from dash import dcc
from dash import html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np

app = dash.Dash()
server = app.server

scaler=MinMaxScaler(feature_range=(0,1))

df_btc = pd.read_csv("./BTC-USD-2023.csv")
df_eth = pd.read_csv("./ETH-USD-2023.csv")
df_ada = pd.read_csv("./ADA-USD-2023.csv")

df_eth["Date"] = pd.to_datetime(df_eth.Date,format="%Y-%m-%d")
df_eth.index=df_eth['Date']

df_btc["Date"] = pd.to_datetime(df_btc.Date,format="%Y-%m-%d")
df_btc.index=df_btc['Date']

df_ada["Date"] = pd.to_datetime(df_ada.Date,format="%Y-%m-%d")
df_ada.index=df_ada['Date']

data_btc = df_btc.sort_index(ascending = True,axis = 0)
new_data_btc = pd.DataFrame(index = range(0,len(data_btc)), columns = ['Date','Close'])
for i in range(0,len(data_btc)):
    new_data_btc["Date"][i]=data_btc['Date'][i]
    new_data_btc["Close"][i]=data_btc["Close"][i]

data_eth = df_eth.sort_index(ascending = True,axis = 0)
new_data_eth = pd.DataFrame(index = range(0,len(data_eth)), columns = ['Date','Close'])
for i in range(0,len(data_eth)):
    new_data_eth["Date"][i]=data_eth['Date'][i]
    new_data_eth["Close"][i]=data_eth["Close"][i]

data_ada = df_ada.sort_index(ascending = True,axis = 0)
new_data_ada = pd.DataFrame(index = range(0,len(data_ada)), columns = ['Date','Close'])
for i in range(0,len(data_ada)):
    new_data_ada["Date"][i]=data_ada['Date'][i]
    new_data_ada["Close"][i]=data_ada["Close"][i]

new_data_btc.index=new_data_btc.Date
new_data_btc.drop("Date",axis=1,inplace=True)
dataset_btc=new_data_btc.values

new_data_eth.index=new_data_eth.Date
new_data_eth.drop("Date",axis=1,inplace=True)
dataset_eth=new_data_eth.values

new_data_ada.index=new_data_ada.Date
new_data_ada.drop("Date",axis=1,inplace=True)
dataset_ada=new_data_ada.values

n_btc = int(dataset_btc.shape[0]/3) * 2
n_eth = int(dataset_eth.shape[0]/3) * 2
n_ada = int(dataset_ada.shape[0]/3) * 2

train_btc=dataset_btc[0:n_btc,:]
valid_btc=dataset_btc[n_btc:,:]

train_eth=dataset_eth[0:n_eth,:]
valid_eth=dataset_eth[n_eth:,:]

train_ada=dataset_ada[0:n_ada,:]
valid_ada=dataset_ada[n_ada:,:]

scaler_btc=MinMaxScaler(feature_range=(0,1))
scaler_eth=MinMaxScaler(feature_range=(0,1))
scaler_ada=MinMaxScaler(feature_range=(0,1))

scaled_data_btc=scaler_btc.fit_transform(dataset_btc)
scaled_data_eth=scaler_eth.fit_transform(dataset_eth)
scaled_data_ada=scaler_ada.fit_transform(dataset_ada)

x_train_btc,y_train_btc=[],[]
x_train_eth,y_train_eth=[],[]
x_train_ada,y_train_ada=[],[]

for i in range(60,len(train_btc)):
    x_train_btc.append(scaled_data_btc[i-60:i,0])
    y_train_btc.append(scaled_data_btc[i,0])

for i in range(60,len(train_eth)):
    x_train_eth.append(scaled_data_eth[i-60:i,0])
    y_train_eth.append(scaled_data_eth[i,0])

for i in range(60,len(train_ada)):
    x_train_ada.append(scaled_data_ada[i-60:i,0])
    y_train_ada.append(scaled_data_ada[i,0])
    
x_train_btc,y_train_btc=np.array(x_train_btc),np.array(y_train_btc)
x_train_btc=np.reshape(x_train_btc,(x_train_btc.shape[0],x_train_btc.shape[1],1))

x_train_eth,y_train_eth=np.array(x_train_eth),np.array(y_train_eth)
x_train_eth=np.reshape(x_train_eth,(x_train_eth.shape[0],x_train_eth.shape[1],1))

x_train_ada,y_train_ada=np.array(x_train_ada),np.array(y_train_ada)
x_train_ada=np.reshape(x_train_ada,(x_train_ada.shape[0],x_train_ada.shape[1],1))

model_btc=load_model("btc_model.h5")
model_eth=load_model("eth_model.h5")
model_ada=load_model("ada_model.h5")

inputs_btc=new_data_btc[len(new_data_btc)-len(valid_btc)-60:].values
inputs_btc=inputs_btc.reshape(-1,1)
inputs_btc=scaler_btc.transform(inputs_btc)

inputs_eth=new_data_eth[len(new_data_eth)-len(valid_eth)-60:].values
inputs_eth=inputs_eth.reshape(-1,1)
inputs_eth=scaler_eth.transform(inputs_eth)

inputs_ada=new_data_ada[len(new_data_ada)-len(valid_ada)-60:].values
inputs_ada=inputs_ada.reshape(-1,1)
inputs_ada=scaler_ada.transform(inputs_ada)

X_test_btc=[]
for i in range(60,inputs_btc.shape[0]):
    X_test_btc.append(inputs_btc[i-60:i,0])
X_test_btc=np.array(X_test_btc)
X_test_btc=np.reshape(X_test_btc,(X_test_btc.shape[0],X_test_btc.shape[1],1))

X_test_eth=[]
for i in range(60,inputs_eth.shape[0]):
    X_test_eth.append(inputs_eth[i-60:i,0])
X_test_eth=np.array(X_test_eth)
X_test_eth=np.reshape(X_test_eth,(X_test_eth.shape[0],X_test_eth.shape[1],1))

X_test_ada=[]
for i in range(60,inputs_ada.shape[0]):
    X_test_ada.append(inputs_ada[i-60:i,0])
X_test_ada=np.array(X_test_ada)
X_test_ada=np.reshape(X_test_ada,(X_test_ada.shape[0],X_test_ada.shape[1],1))


closing_price_btc=model_btc.predict(X_test_btc)
closing_price_btc=scaler_btc.inverse_transform(closing_price_btc)

closing_price_eth=model_eth.predict(X_test_eth)
closing_price_eth=scaler_eth.inverse_transform(closing_price_eth)

closing_price_ada=model_ada.predict(X_test_ada)
closing_price_ada=scaler_ada.inverse_transform(closing_price_ada)

train_btc=new_data_btc[:n_btc]
valid_btc=new_data_btc[n_btc:]
valid_btc['Predictions']=closing_price_btc

train_eth=new_data_eth[:n_eth]
valid_eth=new_data_eth[n_eth:]
valid_eth['Predictions']=closing_price_eth

train_ada=new_data_ada[:n_ada]
valid_ada=new_data_ada[n_ada:]
valid_ada['Predictions']=closing_price_ada

df= pd.read_csv("./coin_data.csv")

app.layout = html.Div([
    html.H1("Coin Price Analysis Dashboard", style={"textAlign": "center"}),

    dcc.Tabs(
        id="tabs", children=[
            dcc.Tab(label='BTC-USD 20-23 Data',children=[
			html.Div([
				html.H2("Actual closing price",style={"textAlign": "center"}),
				dcc.Graph(
					id="Actual Data",
					figure={
						"data":[
							go.Scatter(
								x=train_btc.index,
								y=valid_btc["Close"],
								mode='markers'
							)

						],
						"layout":go.Layout(
							title='scatter plot',
							xaxis={'title':'Date'},
							yaxis={'title':'Closing Rate'}
						)
					}

				),
				html.H2("LSTM Predicted closing price",style={"textAlign": "center"}),
				dcc.Graph(
					id="Predicted Data",
					figure={
						"data":[
							go.Scatter(
								x=train_btc.index,
								y=valid_btc["Predictions"],
								mode='markers'
							)

						],
						"layout":go.Layout(
							title='scatter plot',
							xaxis={'title':'Date'},
							yaxis={'title':'Closing Rate'}
						)
					}

				)				
			])       
        ]),

        dcc.Tab(label='ETH-USD 20-23 Data',children=[
			html.Div([
				html.H2("Actual closing price",style={"textAlign": "center"}),
				dcc.Graph(
					id="Actual Data",
					figure={
						"data":[
							go.Scatter(
								x=train_eth.index,
								y=valid_eth["Close"],
								mode='markers'
							)

						],
						"layout":go.Layout(
							title='scatter plot',
							xaxis={'title':'Date'},
							yaxis={'title':'Closing Rate'}
						)
					}

				),
				html.H2("LSTM Predicted closing price",style={"textAlign": "center"}),
				dcc.Graph(
					id="Predicted Data",
					figure={
						"data":[
							go.Scatter(
								x=train_eth.index,
								y=valid_eth["Predictions"],
								mode='markers'
							)

						],
						"layout":go.Layout(
							title='scatter plot',
							xaxis={'title':'Date'},
							yaxis={'title':'Closing Rate'}
						)
					}

				)				
			])       
        ]),

        dcc.Tab(label='ADA-USD 20-23 Data',children=[
			html.Div([
				html.H2("Actual closing price",style={"textAlign": "center"}),
				dcc.Graph(
					id="Actual Data",
					figure={
						"data":[
							go.Scatter(
								x=train_ada.index,
								y=valid_ada["Close"],
								mode='markers'
							)

						],
						"layout":go.Layout(
							title='scatter plot',
							xaxis={'title':'Date'},
							yaxis={'title':'Closing Rate'}
						)
					}

				),
				html.H2("LSTM Predicted closing price",style={"textAlign": "center"}),
				dcc.Graph(
					id="Predicted Data",
					figure={
						"data":[
							go.Scatter(
								x=train_ada.index,
								y=valid_ada["Predictions"],
								mode='markers'
							)

						],
						"layout":go.Layout(
							title='scatter plot',
							xaxis={'title':'Date'},
							yaxis={'title':'Closing Rate'}
						)
					}

				)				
			])       
        ]),
        


        dcc.Tab(label='Coin 22-23 Data', children=[
            html.Div([
                html.H1("High vs Lows", 
                        style={'textAlign': 'center'}),
              
                dcc.Dropdown(id='my-dropdown',
                             options=[{'label': 'BTC-USD', 'value': 'BTCUSD'},
                                      {'label': 'ETH-USD','value': 'ETHUSD'}, 
                                      {'label': 'ADA-USD', 'value': 'ADAUSD'}], 
                             multi=True,value=['BTCUSD'],
                             style={"display": "block", "margin-left": "auto", 
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='highlow'),

                html.H1("Volume", style={'textAlign': 'center'}),
                dcc.Dropdown(id='my-dropdown2',
                             options=[{'label': 'BTC-USD', 'value': 'BTCUSD'},
                                      {'label': 'ETH-USD','value': 'ETHUSD'}, 
                                      {'label': 'ADA-USD', 'value': 'ADAUSD'}], 
                             multi=True,value=['BTCUSD'],
                             style={"display": "block", "margin-left": "auto", 
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='volume')

            ], className="container"),
        ])
    ])

])

@app.callback(Output('highlow', 'figure'),
              [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown):
    dropdown = {"BTCUSD": "BTC-USD","ETHUSD": "ETH-USD", "ADAUSD": "ADA-USD"}
    trace1 = []
    trace2 = []
    for stock in selected_dropdown:
        trace1.append(
          go.Scatter(x=df[df["Stock"] == stock]["Date"],
                     y=df[df["Stock"] == stock]["High"],
                     mode='lines', opacity=0.7, 
                     name=f'High {dropdown[stock]}',textposition='bottom center'))
        trace2.append(
          go.Scatter(x=df[df["Stock"] == stock]["Date"],
                     y=df[df["Stock"] == stock]["Low"],
                     mode='lines', opacity=0.6,
                     name=f'Low {dropdown[stock]}',textposition='bottom center'))
    traces = [trace1, trace2]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            title=f"High and Low Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Price (USD)"})}
    return figure


@app.callback(Output('volume', 'figure'), [Input('my-dropdown2', 'value')])
def update_graph(selected_dropdown_value):
    dropdown = {"BTCUSD": "BTC-USD","ETHUSD": "ETH-USD","ADAUSD": "ADA-USD"}
    trace1 = []
    for stock in selected_dropdown_value:
        trace1.append(
          go.Scatter(x=df[df["Stock"] == stock]["Date"],
                     y=df[df["Stock"] == stock]["Volume"],
                     mode='lines', opacity=0.7,
                     name=f'Volume {dropdown[stock]}', textposition='bottom center'))
    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data, 
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            title=f"Volume for {', '.join(str(dropdown[i]) for i in selected_dropdown_value)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M',
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Volume"})}
    return figure



if __name__=='__main__':
	app.run_server(debug=True)