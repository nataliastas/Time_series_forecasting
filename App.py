from typing import Any
import plotly.express as px
import pandas as pd
from dash import Dash, html, dcc
from dash import Input, Output, callback
from dash.dependencies import Input, Output
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
import numpy as np
import pickle
import plotly.graph_objects as go
import os
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.feature_selection import chi2, f_classif, f_regression
from dash import dash_table
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.ar_model import AutoReg
import tensorflow as tf
from tensorflow import keras
import statsmodels.api as sm


app = Dash(__name__, assets_folder="../assets")
colors = {"background": "#F5CCB0", "text": "#F57C00"}

dir = os.path.dirname(__file__)
path_1 = os.path.join(dir, "train.csv")
path_2 = os.path.join(dir, 'raw_sales.csv')
path_3 = os.path.join(dir, 'Wyniki.xlsx')

def import_scores(path_3):
    scores_1 = pd.read_excel(path_3, sheet_name='Arkusz1')
    scores_2 = pd.read_excel(path_3, sheet_name='Arkusz2')
    return scores_1, scores_2

scores_1, scores_2 = import_scores(path_3)

def import_data_1(path):
    data_train = pd.read_csv(path)
    data_train_clean = data_train[["Store", "Date", "Weekly_Sales"]]
    data_train_ready = data_train_clean.groupby(["Date"], as_index=False).sum()
    data_train_ready = data_train_ready[["Date", "Weekly_Sales"]]
    label_encoder = preprocessing.LabelEncoder()
    data_train["IsHoliday"] = label_encoder.fit_transform(data_train["IsHoliday"])
    data_train_clean_many = data_train[["Date", "Weekly_Sales", "IsHoliday"]]
    data_train_ready_many = data_train_clean_many.groupby(
        ["Date", "IsHoliday"], as_index=False
    ).sum()
    return data_train_ready, data_train_ready_many

data_train_ready_one_feature, data_train_ready_many_features = import_data_1(path_1)

def import_data_2(path):
    data = pd.read_csv(path)
    data['datesold'] = pd.to_datetime(data['datesold'])
    data.drop_duplicates(subset = ['datesold'], inplace = True)
    data = data[data['price'] <= 2000000]
    data = data.sort_values(by = ['datesold'])
    data_one = data[['datesold', 'price']]
    data_many = data.reset_index(drop=True)
    encoder = OneHotEncoder(sparse_output=False)
    one_hot_encoded = encoder.fit_transform(data_many[['propertyType']])
    one_hot_df = pd.DataFrame(one_hot_encoded, columns = encoder.get_feature_names_out(['propertyType']))
    data_many = pd.concat([data_many, one_hot_df], axis = 1)
    data_many = data_many.drop(['propertyType'], axis = 1)
    return data_one, data_many

data2_one_feature, data2_many_features = import_data_2(path_2)

def data_ML_one_feature2(data):
    data["datesold"] = pd.to_datetime(data["datesold"])
    data = data.copy()
    data["dayofweek"] = data.datesold.dt.weekday
    data["quarter"] = data.datesold.dt.quarter
    data["month"] = data.datesold.dt.month
    data["year"] = data.datesold.dt.year
    data["dayofyear"] = data.datesold.dt.dayofyear
    data["dayofmonth"] = data.datesold.dt.day
    data["weekofyear"] = data.datesold.dt.isocalendar().week
    train_len = int(0.9 * len(data))
    train = data[:train_len]
    test = data[train_len:]
    X_train = train[
        [
            "dayofweek",
            "quarter",
            "month",
            "year",
            "dayofyear",
            "dayofmonth",
            "weekofyear",
        ]
    ]
    Y_train = train[["price"]]
    X_test = test[
        [
            "dayofweek",
            "quarter",
            "month",
            "year",
            "dayofyear",
            "dayofmonth",
            "weekofyear",
        ]
    ]
    Y_test = test[["price"]]
    return X_train, Y_train, X_test, Y_test, train_len


X_train_one_2, Y_train_one_2, X_test_one_2, Y_test_one_2, train_len_one_2 = data_ML_one_feature2(
    data2_one_feature
)

def data_ML_one_feature(data):
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.copy()
    data["dayofweek"] = data.Date.dt.weekday
    data["quarter"] = data.Date.dt.quarter
    data["month"] = data.Date.dt.month
    data["year"] = data.Date.dt.year
    data["dayofyear"] = data.Date.dt.dayofyear
    data["dayofmonth"] = data.Date.dt.day
    data["weekofyear"] = data.Date.dt.isocalendar().week
    train_len = int(0.9 * len(data))
    train = data[:train_len]
    test = data[train_len:]
    X_train = train[
        [
            "dayofweek",
            "quarter",
            "month",
            "year",
            "dayofyear",
            "dayofmonth",
            "weekofyear",
        ]
    ]
    Y_train = train[["Weekly_Sales"]]
    X_test = test[
        [
            "dayofweek",
            "quarter",
            "month",
            "year",
            "dayofyear",
            "dayofmonth",
            "weekofyear",
        ]
    ]
    Y_test = test[["Weekly_Sales"]]
    return X_train, Y_train, X_test, Y_test, train_len


X_train_one, Y_train_one, X_test_one, Y_test_one, train_len_one = data_ML_one_feature(
    data_train_ready_one_feature
)

def data_ML_many_features2(data):
    data["datesold"] = pd.to_datetime(data["datesold"])
    data = data.copy()
    data["dayofweek"] = data.datesold.dt.weekday
    data["quarter"] = data.datesold.dt.quarter
    data["month"] = data.datesold.dt.month
    data["year"] = data.datesold.dt.year
    data["dayofyear"] = data.datesold.dt.dayofyear
    data["dayofmonth"] = data.datesold.dt.day
    data["weekofyear"] = data.datesold.dt.isocalendar().week
    train_len = int(0.9 * len(data))
    train = data[:train_len]
    test = data[train_len:]
    X_train = train[
        [
            "dayofweek",
            "quarter",
            "month",
            "year",
            "dayofyear",
            "dayofmonth",
            "weekofyear",
            'postcode', 
            'bedrooms', 
            'propertyType_house',
            'propertyType_unit'
        ]
    ]
    Y_train = train[["price"]]
    X_test = test[
        [
            "dayofweek",
            "quarter",
            "month",
            "year",
            "dayofyear",
            "dayofmonth",
            "weekofyear",
            'postcode', 
            'bedrooms', 
            'propertyType_house',
            'propertyType_unit'
        ]
    ]
    Y_test = test[["price"]]
    return X_train, Y_train, X_test, Y_test, train_len


X_train_many_2, Y_train_many_2, X_test_many_2, Y_test_many_2, train_len_many_2 = data_ML_many_features2(data2_many_features)

def data_ML_many_features(data):
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.copy()
    data["dayofweek"] = data.Date.dt.weekday
    data["quarter"] = data.Date.dt.quarter
    data["month"] = data.Date.dt.month
    data["year"] = data.Date.dt.year
    data["dayofyear"] = data.Date.dt.dayofyear
    data["dayofmonth"] = data.Date.dt.day
    data["weekofyear"] = data.Date.dt.isocalendar().week
    train_len = int(0.9 * len(data))
    train = data[:train_len]
    test = data[train_len:]
    X_train = train[
        [
            "dayofweek",
            "quarter",
            "month",
            "year",
            "dayofyear",
            "dayofmonth",
            "weekofyear",
            "IsHoliday",
        ]
    ]
    Y_train = train[["Weekly_Sales"]]
    X_test = test[
        [
            "dayofweek",
            "quarter",
            "month",
            "year",
            "dayofyear",
            "dayofmonth",
            "weekofyear",
            "IsHoliday",
        ]
    ]
    Y_test = test[["Weekly_Sales"]]
    return X_train, Y_train, X_test, Y_test, train_len


X_train_many, Y_train_many, X_test_many, Y_test_many, train_len_many = data_ML_many_features(data_train_ready_many_features)

def display_time_series(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Weekly_Sales"]))
    return fig

def display_time_series2(data2_one_feature):    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data2_one_feature['datesold'], y=data2_one_feature['price']))
    return fig

def plot_predict(data, train_len, pred, output, input):
    fig2 = go.Figure()
    fig2.add_trace(
        go.Line(
            x=input[train_len : len(data)].reset_index(drop=True),
            y=output[train_len : len(data)].reset_index(drop=True),
            name="test",
            fill = None
        )
    )
    fig2.add_trace(
        go.Line(
            x=input[:train_len].reset_index(drop=True),
            y=output[:train_len].reset_index(drop=True),
            name="train",
            fill = None
        )
    )
    fig2.add_trace(
        go.Line(
            x=input[train_len : len(data)].reset_index(drop=True),
            y=pred,
            name="prediction",
            fill = None
        )
    )
    return fig2

def AR_model(data, output, lags):
    train_len = int(0.9 * len(data))
    train = output[:train_len]
    ar_model = AutoReg(train, lags=lags).fit()
    pred = ar_model.predict(start=train_len, end=len(data), dynamic=False)
    pred = pred.reset_index(drop = True)
    return pred, train_len


def import_data_with_many_features(path):
    data_train = pd.read_csv(path)
    label_encoder = preprocessing.LabelEncoder()
    data_train["IsHoliday"] = label_encoder.fit_transform(data_train["IsHoliday"])
    data_train_clean = data_train[["Date", "Weekly_Sales", "IsHoliday"]]
    data_train_ready = data_train_clean.groupby(
        ["Date", "IsHoliday"], as_index=False
    ).sum()
    data_train_ready
    return data_train_ready

def VAR_method(data, name):
    train_len = int(0.9 * len(data))
    train = data[: int(0.9 * (len(data)))]
    test = data[int(0.9 * (len(data))) :]
    train.index = train[name]
    train = train.drop([name], axis=1)
    test.index = test[name]
    test = test.drop([name], axis=1)
    var_model = VAR(np.asarray(train))
    optimal_lags = var_model.select_order()
    lag_order = optimal_lags.selected_orders["bic"]
    results = var_model.fit(lag_order )
    forecast_input = train.values[-lag_order :]
    pred = results.forecast(forecast_input, steps=len(test))
    pred = pred[:, lag_order]
    return pred, train_len

def VAR_method2(data, name, counter):
    train_len = int(0.9 * len(data))
    train = data[: int(0.9 * (len(data)))]
    test = data[int(0.9 * (len(data))) :]
    train.index = train[name]
    train = train.drop([name], axis=1)
    test.index = test[name]
    test = test.drop([name], axis=1)
    var_model = VAR(np.asarray(train))
    results = var_model.fit(1)
    forecast_input = train.values[-1:]
    pred = results.forecast(forecast_input, steps=len(test))
    pred = pred[:, counter]
    return pred, train_len
predictions_VAR_2, train_len_2 = VAR_method2(data2_many_features, 'datesold', 4)

def MA_method(data, name):
    train_len = int(0.9 * len(data))
    train = data[: int(0.9 * (len(data)))]
    test = data[int(0.9 * (len(data))) :]
    train.index = train[name]
    train = train.drop([name], axis=1)
    test.index = test[name]
    test = test.drop([name], axis=1)
    model = ARIMA(np.asarray(train), order=(0, 0, 5))
    model_fit = model.fit()
    pred_ma = model_fit.get_forecast(steps=len(test))
    pred_ma_series = pd.Series(pred_ma.predicted_mean, index=test.index)
    pred = pred_ma_series.values
    return pred, train_len

def data_ARMA(data, name):
    train_len = int(0.9 * len(data))
    train = data[: int(0.9 * (len(data)))]
    test = data[int(0.9 * (len(data))) :]
    train.index = train[name]
    train = train.drop([name], axis=1)
    test.index = test[name]
    test = test.drop([name], axis=1)
    return train, test, train_len

train_ARMA, test_ARMA, train_len_ARMA = data_ARMA(data_train_ready_one_feature, 'Date')
train_ARMA_2, test_ARMA_2, train_len_2_ARMA = data_ARMA(data2_one_feature, 'datesold')

def ARMA_method(train, test, order):
    model = ARIMA(np.asarray(train), order=order)
    model_fit = model.fit()
    pred_ma = model_fit.get_forecast(steps=len(test))
    pred_ma_series = pd.Series(pred_ma.predicted_mean, index=test.index)
    pred = pred_ma_series.values
    return pred

def ARIMA_method(train, test, order):
    model = ARIMA(np.asarray(train), order=order)
    model_fit = model.fit()
    pred_ma = model_fit.get_forecast(steps=len(test))
    pred_ma_series = pd.Series(pred_ma.predicted_mean, index=test.index)
    pred = pred_ma_series.values
    return pred

def SARIMA_method(train, test, order1, order2):
    model = sm.tsa.statespace.SARIMAX(
        np.asarray(train), order=order1, seasonal_order=order2
    )
    model_fit = model.fit()
    pred_ma = model_fit.get_forecast(steps=len(test))
    pred_ma_series = pd.Series(pred_ma.predicted_mean, index=test.index)
    pred = pred_ma_series.values
    return pred

def Decision_Tree_predict(name, X_test, Y_test):
    pipe = pickle.load(open(name, "rb"))
    predictions = pipe.predict(X_test)
    rmse = float(format(np.sqrt(mean_squared_error(Y_test, predictions)), ".3f"))
    return predictions

def Random_Forest_predict(name, X_test, Y_test):
    pipe = pickle.load(open(name, "rb"))
    predictions = pipe.predict(X_test)
    rmse = float(format(np.sqrt(mean_squared_error(Y_test, predictions)), ".3f"))
    return predictions

def XGB_method_predict(name, X_test, Y_test):
    pipe = pickle.load(open(name, "rb"))
    predictions = pipe.predict(X_test)
    rmse = float(format(np.sqrt(mean_squared_error(Y_test, predictions)), ".3f"))
    return predictions

def data_neural_many_features(data):
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.copy()
    data["dayofweek"] = data.Date.dt.weekday
    data["quarter"] = data.Date.dt.quarter
    data["month"] = data.Date.dt.month
    data["year"] = data.Date.dt.year
    data["dayofyear"] = data.Date.dt.dayofyear
    data["dayofmonth"] = data.Date.dt.day
    data["weekofyear"] = data.Date.dt.isocalendar().week
    train_len = int(0.9 * len(data))
    train = data[:train_len]
    test = data[train_len:]
    validation_len = int(0.1 * len(train))
    validation = train[:validation_len]
    train = train[validation_len:]
    X_train = train[
        [
            "dayofweek",
            "quarter",
            "month",
            "year",
            "dayofyear",
            "dayofmonth",
            "weekofyear",
            "IsHoliday",
        ]
    ]
    Y_train = train[["Weekly_Sales"]]
    X_test = test[
        [
            "dayofweek",
            "quarter",
            "month",
            "year",
            "dayofyear",
            "dayofmonth",
            "weekofyear",
            "IsHoliday",
        ]
    ]
    Y_test = test[["Weekly_Sales"]]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    return X_train, Y_train, X_test, Y_test, train_len

X_train, Y_train, X_test, Y_test, train_len_neural = data_neural_many_features(
    data_train_ready_many_features
)

def data_neural_many_features2(data):
    data["datesold"] = pd.to_datetime(data["datesold"])
    data = data.copy()
    data["dayofweek"] = data.datesold.dt.weekday
    data["quarter"] = data.datesold.dt.quarter
    data["month"] = data.datesold.dt.month
    data["year"] = data.datesold.dt.year
    data["dayofyear"] = data.datesold.dt.dayofyear
    data["dayofmonth"] = data.datesold.dt.day
    data["weekofyear"] = data.datesold.dt.isocalendar().week
    train_len = int(0.9 * len(data))
    train = data[:train_len]
    test = data[train_len:]
    X_train = train[
        [
            "dayofweek",
            "quarter",
            "month",
            "year",
            "dayofyear",
            "dayofmonth",
            "weekofyear",
            'postcode', 
            'bedrooms', 
            'propertyType_house',
            'propertyType_unit'
        ]
    ]
    Y_train = train[["price"]]
    X_test = test[
        [
            "dayofweek",
            "quarter",
            "month",
            "year",
            "dayofyear",
            "dayofmonth",
            "weekofyear",
            'postcode', 
            'bedrooms', 
            'propertyType_house',
            'propertyType_unit'
        ]
    ]
    Y_test = test[["price"]]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    return X_train, Y_train, X_test, Y_test, train_len


X_train2, Y_train2, X_test2, Y_test2, train_len_neural2 = data_neural_many_features2(
    data2_many_features
)

def neural_networks_predict(name, X_train, Y_train, X_test, Y_test):
    model = pickle.load(open(name, "rb"))
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    pred = y_test_pred[:, 0]
    rmse_train = float(
        format(np.sqrt(mean_squared_error(Y_train, y_train_pred)), ".3f")
    )
    rmse_test = float(format(np.sqrt(mean_squared_error(Y_test, y_test_pred)), ".3f"))
    return pred

methods_array = np.array(["AR method", "MA method", "VAR method", "ARMA method", "ARIMA method", "SARIMA method", "Decision tree", 
                          "Random Forest", "XGBoost", "MLP", "CNN", "RNN", "LSTM", "GRU"])
data_array = np.array(["Dane sprzedażowe - Walmart", "Dane o cenach nieruchomości"])
count_features_array = np.array(["One feature", "Many features"])



#App
app.layout = html.Div(
    style={"backgroundColor": colors["background"]},
    children=[
        html.H1(
            children="Aplikacja do porównywania skuteczności metod w prognozowaniu szeregów czasowych",
            style={"textAlign": "center", "color": colors["text"]},
        ),
        html.Div(
            [
                html.H3(
                    children="Wybierz zestaw danych",
                    style={"textAlign": "center", "color": colors["text"]},
                    className="card",
                ),
                dcc.Dropdown(np.unique(data_array), None, id="data-selection"),
            ],
        ),
        html.Div(dcc.Graph(id="chart")),
        html.Div(
            [
                html.H3(
                    children="Tabela z porównaniem skuteczności różnych metod na podstawie wybranych metryk",
                    style={"textAlign": "center", "color": colors["text"]},
                ),
            ]
        ),
        dash_table.DataTable(id = "table", 
            style_cell={'padding': '5px'},
            style_header={
        'fontWeight': 'bold'
    },
            style_cell_conditional=[
        {
            'if': {'column_id': c},
            'textAlign': 'left'
        } for c in ['Rodzaje', 'Metody']
    ],

    style_as_list_view=True,
),
        html.Div(
            [
                html.H3(
                    children="Wybierz metodę, aby zobaczyć wykres dopasowania",
                    style={"textAlign": "center", "color": colors["text"]},
                    className="card",
                ),
                dcc.Dropdown(np.unique(methods_array), None, id="method-selection"),
            ]
        ),
        html.Div(dcc.Graph(id="chart2")),
        html.Div(dcc.Graph(id="chart3")),
    ],
)


@callback(
    Output("chart", "figure"),
    Output("table", "data"),
    Output("table", "columns"),
    Output("chart2", "figure"),
    Output("chart3", "figure"),
    Input("method-selection", "value"),
    Input("data-selection", "value"),
)
def update(

    selected_method_value: str,
    selected_data_value: str,
) -> Any:

    fig2 = px.line()

    if selected_data_value == "Dane sprzedażowe - Walmart":
        fig = display_time_series(data_train_ready_one_feature)
        columns = [{'name': col, 'id': col} for col in scores_1.columns]
        tab = scores_1.to_dict('records')
    elif selected_data_value == "Dane o cenach nieruchomości":
        fig = display_time_series2(data2_one_feature)
        columns = [{'name': col, 'id': col} for col in scores_2.columns]
        tab = scores_2.to_dict('records')
    else:
        columns = [{'name': col, 'id': col} for col in scores_1.columns]
        fig = fig2
        tab = scores_1.to_dict('records')

    if selected_method_value == "AR method":
        if selected_data_value == 'Dane sprzedażowe - Walmart':
            prediction_AR, train_len = AR_model(data_train_ready_one_feature, data_train_ready_one_feature.Weekly_Sales, 5)
            fig3 = plot_predict(data_train_ready_one_feature, train_len, prediction_AR, data_train_ready_one_feature.Weekly_Sales, 
                data_train_ready_one_feature.Date)
            fig4 = fig2
        elif selected_data_value == 'Dane o cenach nieruchomości':
            prediction_AR_2, train_len_2 = AR_model(data2_one_feature, data2_one_feature.price, 5)
            fig3 = plot_predict(data2_one_feature, train_len_2, prediction_AR_2, data2_one_feature.price, data2_one_feature.datesold)
            fig4 = fig2
    elif selected_method_value == 'MA method':
        if selected_data_value == 'Dane sprzedażowe - Walmart':
            pred_MA, train_len = MA_method(data_train_ready_one_feature, 'Date')
            fig3 = plot_predict(data_train_ready_one_feature, train_len, pred_MA, data_train_ready_one_feature.Weekly_Sales, 
                        data_train_ready_one_feature.Date)
            fig4 = fig2
        elif selected_data_value == 'Dane o cenach nieruchomości':
            pred_MA_2, train_len_2 = MA_method(data2_one_feature, 'datesold')
            fig3 = plot_predict(data2_one_feature, train_len_2, pred_MA_2, data2_one_feature.price, 
             data2_one_feature.datesold)
            fig4 = fig2
    elif selected_method_value == 'VAR method':
        if selected_data_value == 'Dane sprzedażowe - Walmart':
            predictions_VAR, train_len = VAR_method(data_train_ready_many_features, 'Date')
            fig4 = plot_predict(data_train_ready_many_features, train_len, predictions_VAR, data_train_ready_many_features.Weekly_Sales,
            data_train_ready_many_features.Date)
            fig3 = fig2
        elif selected_data_value == 'Dane o cenach nieruchomości':
            predictions_VAR_2, train_len_2 = VAR_method2(data2_many_features, 'datesold', 4)
            fig4 = plot_predict(data2_many_features, train_len_2, predictions_VAR_2, data2_many_features.price, data2_many_features.datesold)
            fig3 = fig2
    elif selected_method_value == 'ARMA method':
        if selected_data_value == 'Dane sprzedażowe - Walmart':
            prediction_ARMA = ARMA_method(train_ARMA, test_ARMA, (16,0,4))
            fig3 = plot_predict(data_train_ready_one_feature, train_len_ARMA, prediction_ARMA, data_train_ready_one_feature.Weekly_Sales, 
             data_train_ready_one_feature.Date)
            fig4 = fig2
        elif selected_data_value == 'Dane o cenach nieruchomości':
            prediction_ARMA_2 = ARMA_method(train_ARMA_2, test_ARMA_2, (4,0,5))
            fig3 = plot_predict(data2_one_feature, train_len_2_ARMA, prediction_ARMA_2, data2_one_feature.price, 
             data2_one_feature.datesold)
            fig4 = fig2
    elif selected_method_value == 'ARIMA method':
        if selected_data_value == 'Dane sprzedażowe - Walmart':
            prediction_ARIMA = ARIMA_method(train_ARMA, test_ARMA, (16,0,4))
            fig3 = plot_predict(data_train_ready_one_feature, train_len_ARMA, prediction_ARIMA, data_train_ready_one_feature.Weekly_Sales, 
             data_train_ready_one_feature.Date)
            fig4 = fig2
        elif selected_data_value == 'Dane o cenach nieruchomości':
            prediction_ARIMA_2 = ARIMA_method(train_ARMA_2, test_ARMA_2, (4,0,5))
            fig3 = plot_predict(data2_one_feature, train_len_2_ARMA, prediction_ARIMA_2, data2_one_feature.price, 
             data2_one_feature.datesold)
            fig4 = fig2
    elif selected_method_value == 'SARIMA method':
        if selected_data_value == 'Dane sprzedażowe - Walmart':
            prediction_SARIMA = SARIMA_method(train_ARMA, test_ARMA, (1,0,0), (0, 0, 0, 12))
            fig3 = plot_predict(data_train_ready_one_feature, train_len_ARMA, prediction_SARIMA, data_train_ready_one_feature.Weekly_Sales, 
             data_train_ready_one_feature.Date)
            fig4 = fig2
        elif selected_data_value == 'Dane o cenach nieruchomości':
            prediction_SARIMA_2 = SARIMA_method(train_ARMA_2, test_ARMA_2, (0,0,2), (0, 0, 1, 12))
            fig3 = plot_predict(data2_one_feature, train_len_2_ARMA, prediction_SARIMA_2, data2_one_feature.price, 
             data2_one_feature.datesold)
            fig4 = fig2
    elif selected_method_value == 'Decision tree':
        if selected_data_value == 'Dane sprzedażowe - Walmart':
            prediction_DT_one = Decision_Tree_predict("model_DT_one.pkl", X_test_one, Y_test_one)
            fig3 = plot_predict(data_train_ready_one_feature, train_len_one, prediction_DT_one, data_train_ready_one_feature.Weekly_Sales, 
             data_train_ready_one_feature.Date)
            prediction_DT_many = Decision_Tree_predict("model_DT_many.pkl", X_test_many, Y_test_many)
            fig4 = plot_predict(data_train_ready_many_features, train_len_many, prediction_DT_many, data_train_ready_many_features.Weekly_Sales, 
             data_train_ready_many_features.Date)
        elif selected_data_value == 'Dane o cenach nieruchomości':
            prediction_DT_one_2 = Decision_Tree_predict("model_DT_one_2.pkl", X_test_one_2, Y_test_one_2)
            fig3 = plot_predict(data2_one_feature, train_len_one_2, prediction_DT_one_2, data2_one_feature.price, 
             data2_one_feature.datesold)
            prediction_DT_many_2 = Decision_Tree_predict("model_DT_many_2.pkl", X_test_many_2, Y_test_many_2)
            fig4 = plot_predict(data2_many_features, train_len_many_2, prediction_DT_many_2, data2_many_features.price, 
             data2_many_features.datesold)
    elif selected_method_value == 'Random Forest':
        if selected_data_value == 'Dane sprzedażowe - Walmart':
            prediction_RF_one = Random_Forest_predict("model_FR_one.pkl", X_test_one, Y_test_one)
            prediction_RF_many = Random_Forest_predict("model_FR_many.pkl", X_test_many, Y_test_many)
            fig3 = plot_predict(data_train_ready_one_feature, train_len_one, prediction_RF_one, data_train_ready_one_feature.Weekly_Sales, 
             data_train_ready_one_feature.Date)
            fig4 = plot_predict(data_train_ready_many_features, train_len_many, prediction_RF_many, data_train_ready_many_features.Weekly_Sales, 
             data_train_ready_many_features.Date)
        elif selected_data_value == 'Dane o cenach nieruchomości':
            prediction_RF_one_2 = Random_Forest_predict("model_FR_one_2.pkl", X_test_one_2, Y_test_one_2)
            prediction_RF_many_2 = Random_Forest_predict("model_FR_many_2.pkl", X_test_many_2, Y_test_many_2)
            fig3 = plot_predict(data2_one_feature, train_len_one_2, prediction_RF_one_2, data2_one_feature.price, 
             data2_one_feature.datesold)
            fig4 = plot_predict(data2_many_features, train_len_many_2, prediction_RF_many_2, data2_many_features.price, 
             data2_many_features.datesold)
    elif selected_method_value == 'XGBoost':
        if selected_data_value == 'Dane sprzedażowe - Walmart':
            prediction_XGB_one = XGB_method_predict("model_XGB_one.pkl", X_test_one, Y_test_one)
            prediction_XGB_many = XGB_method_predict("model_XGB_many.pkl", X_test_many, Y_test_many)
            fig3 = plot_predict(data_train_ready_one_feature, train_len_one, prediction_XGB_one, data_train_ready_one_feature.Weekly_Sales, 
             data_train_ready_one_feature.Date)
            fig4 = plot_predict(data_train_ready_many_features, train_len_many, prediction_XGB_many, data_train_ready_many_features.Weekly_Sales, 
             data_train_ready_many_features.Date)
        elif selected_data_value == 'Dane o cenach nieruchomości':
            prediction_XGB_one_2 = XGB_method_predict("model_XGB_one_2.pkl", X_test_one_2, Y_test_one_2)
            prediction_XGB_many_2 = XGB_method_predict("model_XGB_many_2.pkl", X_test_many_2, Y_test_many_2)
            fig3 = plot_predict(data2_one_feature, train_len_one_2, prediction_XGB_one_2, data2_one_feature.price, 
             data2_one_feature.datesold)
            fig4 = plot_predict(data2_many_features, train_len_many_2, prediction_XGB_many_2, data2_many_features.price, 
             data2_many_features.datesold)
    elif selected_method_value == 'MLP':
        if selected_data_value == 'Dane sprzedażowe - Walmart':
            MLP_prediction = neural_networks_predict("model_MLP.pkl", X_train, Y_train, X_test, Y_test)
            fig4 = plot_predict(data_train_ready_many_features, train_len_neural, MLP_prediction, data_train_ready_many_features.Weekly_Sales, 
             data_train_ready_many_features.Date)
            fig3 = fig2
        elif selected_data_value == 'Dane o cenach nieruchomości':
            MLP_prediction2 = neural_networks_predict("model_MLP_2.pkl", X_train2, Y_train2, X_test2, Y_test2)
            fig4 = plot_predict(data2_many_features, train_len_neural2, MLP_prediction2, data2_many_features.price, 
             data2_many_features.datesold)
            fig3 = fig2
    elif selected_method_value == 'CNN':
        if selected_data_value == 'Dane sprzedażowe - Walmart':
            CNN_prediction = neural_networks_predict("model_CNN.pkl", X_train, Y_train, X_test, Y_test)
            fig4 = plot_predict(data_train_ready_many_features, train_len_neural, CNN_prediction, data_train_ready_many_features.Weekly_Sales, 
             data_train_ready_many_features.Date)
            fig3 = fig2
        elif selected_data_value == 'Dane o cenach nieruchomości':
            CNN_prediction2 = neural_networks_predict("model_CNN_2.pkl", X_train2, Y_train2, X_test2, Y_test2)
            fig4 = plot_predict(data2_many_features, train_len_neural2, CNN_prediction2, data2_many_features.price, 
             data2_many_features.datesold)
            fig3 = fig2
    elif selected_method_value == 'RNN':
        if selected_data_value == 'Dane sprzedażowe - Walmart':
            RNN_prediction = neural_networks_predict("model_RNN.pkl", X_train, Y_train, X_test, Y_test)
            fig4 = plot_predict(data_train_ready_many_features, train_len_neural, RNN_prediction, data_train_ready_many_features.Weekly_Sales, 
             data_train_ready_many_features.Date)
            fig3 = fig2
        elif selected_data_value == 'Dane o cenach nieruchomości':
            RNN_prediction2 = neural_networks_predict("model_RNN_2.pkl", X_train2, Y_train2, X_test2, Y_test2)
            fig4 = plot_predict(data2_many_features, train_len_neural2, RNN_prediction2, data2_many_features.price, 
             data2_many_features.datesold)
            fig3 = fig2
    elif selected_method_value == 'LSTM':
        if selected_data_value == 'Dane sprzedażowe - Walmart':
            LSTM_prediction = neural_networks_predict("model_LSTM.pkl", X_train, Y_train, X_test, Y_test)
            fig4 = plot_predict(data_train_ready_many_features, train_len_neural, LSTM_prediction, data_train_ready_many_features.Weekly_Sales, 
             data_train_ready_many_features.Date)
            fig3 = fig2
        elif selected_data_value == 'Dane o cenach nieruchomości':
            LSTM_prediction2 = neural_networks_predict("model_LSTM_2.pkl", X_train2, Y_train2, X_test2, Y_test2)
            fig4 = plot_predict(data2_many_features, train_len_neural2, LSTM_prediction2, data2_many_features.price, 
             data2_many_features.datesold)
            fig3 = fig2
    elif selected_method_value == 'GRU':
        if selected_data_value == 'Dane sprzedażowe - Walmart':
            GRU_prediction = neural_networks_predict("model_GRU.pkl", X_train, Y_train, X_test, Y_test)
            fig4 = plot_predict(data_train_ready_many_features, train_len_neural, GRU_prediction, data_train_ready_many_features.Weekly_Sales, 
             data_train_ready_many_features.Date)
            fig3 = fig2
        elif selected_data_value == 'Dane o cenach nieruchomości':
            GRU_prediction2 = neural_networks_predict("model_GRU_2.pkl", X_train2, Y_train2, X_test2, Y_test2)
            fig4 = plot_predict(data2_many_features, train_len_neural2, GRU_prediction2, data2_many_features.price, 
             data2_many_features.datesold)
            fig3 = fig2
    else:
        fig3 = fig2
        fig4 = fig2

    return  fig, tab, columns, fig3, fig4


if __name__ == "__main__":
    app.run_server(debug=True)