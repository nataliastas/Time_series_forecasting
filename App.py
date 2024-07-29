from typing import Any
import plotly.express as px
import pandas as pd
from dash import Dash, html, dcc
from dash import Input, Output, callback
from dash.dependencies import Input, Output
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

app = Dash(__name__, assets_folder="../assets")
colors = {"background": "#F5CCB0", "text": "#F57C00"}

dir = os.path.dirname(__file__)
path_1 = os.path.join(dir, "raw_sales.csv")
path_2 = os.path.join(dir, 'train.csv')

def import_data_1(path):
    data_train = pd.read_csv(path)
    data_train_clean = data_train[["Store", "Date", "Weekly_Sales"]]
    data_train_ready = data_train_clean.groupby(["Date"], as_index=False).sum()
    data_train_ready = data_train_ready[["Date", "Weekly_Sales"]]
    label_encoder = preprocessing.LabelEncoder()
    data_train["IsHoliday"] = label_encoder.fit_transform(data_train["IsHoliday"])
    data_train_clean_many = data_train[["Store", "Date", "Weekly_Sales", "IsHoliday"]]
    data_train_ready_many = data_train_clean_many.groupby(
        ["Date", "Store", "IsHoliday"], as_index=False
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
    data_many = pd.concat([data2_many_features, one_hot_df], axis = 1)
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
            "Store",
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
            "Store",
            "IsHoliday",
        ]
    ]
    Y_test = test[["Weekly_Sales"]]
    return X_train, Y_train, X_test, Y_test, train_len


X_train_many, Y_train_many, X_test_many, Y_test_many, train_len_many = data_ML_many_features(data_train_ready_many_features)

def display_time_series(data):
    sns.lineplot(x=data["Date"], y=data["Weekly_Sales"], marker="o")
    plt.tight_layout()
    plt.show()

def display_time_series2(data2_one_feature):    
    sns.lineplot(x=data2_one_feature['datesold'], y=data2_one_feature['price'])
    plt.tight_layout()
    plt.show()

def plot_predict(data, train_len, pred, output, input):
    f, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,8))
    sns.lineplot(
        x=input[train_len : len(data)],
        y=output[train_len : len(data)],
        marker = 'o',
        label="test",
        color="grey",
    )
    sns.lineplot(
        x=input[:train_len],
        y=output[:train_len],
        marker = 'o',
        label="train",
    )
    sns.lineplot(x=input[train_len : len(data)].reset_index(drop=True), y=pred, marker = 'o', label="pred")
    plt.tight_layout()
    plt.show()

def AR_model(data, output, lags):
    train_len = int(0.9 * len(data))
    train = output[:train_len]
    ar_model = AutoReg(train, lags=lags).fit()
    # print(ar_model.summary())
    pred = ar_model.predict(start=train_len, end=len(data), dynamic=False)
    pred = pred.reset_index(drop = True)
    return pred, train_len


def import_data_with_many_features(path):
    data_train = pd.read_csv(path)
    label_encoder = preprocessing.LabelEncoder()
    data_train["IsHoliday"] = label_encoder.fit_transform(data_train["IsHoliday"])
    data_train_clean = data_train[["Store", "Date", "Weekly_Sales", "IsHoliday"]]
    data_train_ready = data_train_clean.groupby(
        ["Date", "Store", "IsHoliday"], as_index=False
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
    results = var_model.fit(lag_order)
    forecast_input = train.values[-lag_order:]
    print(results.summary())
    pred = results.forecast(forecast_input, steps=len(test))
    pred = pred[:, 2]
    return pred, train_len

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
    print(model_fit.summary())
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

train_ARMA, test_ARMA, train_len = data_ARMA(data_train_ready_one_feature, 'Date')
train_ARMA_2, test_ARMA_2, train_len_2 = data_ARMA(data2_one_feature, 'datesold')

def ARMA_method(train, test, order):
    model = ARIMA(np.asarray(train), order=order)
    model_fit = model.fit()
    print(model_fit.summary())
    pred_ma = model_fit.get_forecast(steps=len(test))
    pred_ma_series = pd.Series(pred_ma.predicted_mean, index=test.index)
    pred = pred_ma_series.values
    return pred

def ARIMA_method(train, test, order):
    model = ARIMA(np.asarray(train), order=order)
    model_fit = model.fit()
    print(model_fit.summary())
    pred_ma = model_fit.get_forecast(steps=len(test))
    pred_ma_series = pd.Series(pred_ma.predicted_mean, index=test.index)
    pred = pred_ma_series.values
    return pred

def SARIMA_method(train, test, order1, order2):
    model = sm.tsa.statespace.SARIMAX(
        np.asarray(train), order=order1, seasonal_order=order2
    )
    model_fit = model.fit()
    print(model_fit.summary())
    pred_ma = model_fit.get_forecast(steps=len(test))
    pred_ma_series = pd.Series(pred_ma.predicted_mean, index=test.index)
    pred = pred_ma_series.values
    return pred

def Decision_Tree_predict(name, X_test, Y_test):
    pipe = pickle.load(open(name, "rb"))
    predictions = pipe.predict(X_test)
    rmse = float(format(np.sqrt(mean_squared_error(Y_test, predictions)), ".3f"))
    print("\nRMSE: ", rmse)
    return predictions

def Random_Forest_predict(name, X_test, Y_test):
    pipe = pickle.load(open(name, "rb"))
    predictions = pipe.predict(X_test)
    rmse = float(format(np.sqrt(mean_squared_error(Y_test, predictions)), ".3f"))
    print("\nRMSE: ", rmse)
    return predictions

def XGB_method_predict(name, X_test, Y_test):
    pipe = pickle.load(open(name, "rb"))
    predictions = pipe.predict(X_test)
    rmse = float(format(np.sqrt(mean_squared_error(Y_test, predictions)), ".3f"))
    print("\nRMSE: ", rmse)
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
            "Store",
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
            "Store",
            "IsHoliday",
        ]
    ]
    Y_test = test[["Weekly_Sales"]]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    return X_train, Y_train, X_test, Y_test

def neural_networks_predict(name, X_train, Y_train, X_test, Y_test):
    model = pickle.load(open(name, "rb"))
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    pred = y_test_pred[:, 0]
    rmse_train = float(
        format(np.sqrt(mean_squared_error(Y_train, y_train_pred)), ".3f")
    )
    rmse_test = float(format(np.sqrt(mean_squared_error(Y_test, y_test_pred)), ".3f"))
    print(rmse_train)
    print(rmse_test)
    return pred

methods_array = np.array(["AR method", "MA method", "VAR method", "ARMA method", "ARIMA method", "SARIMA method", "Decision tree", 
                          "Random Forest", "XGBoost", "MLP", "CNN", "RNN", "LSTM", "GRU"])
data_array = np.array(["Sales data - Walmart", "House prices"])
count_features_array = np.array(["One feature", "Many features"])


#App
app.layout = html.Div(
    style={"backgroundColor": colors["background"]},
    children=[
        html.H1(
            children="App to predict time series",
            style={"textAlign": "center", "color": colors["text"]},
        ),
        html.Div(
            [
                html.H3(
                    children="Wybierz liczbę sypialni",
                    style={"textAlign": "center", "color": colors["text"]},
                    className="card",
                ),
                dcc.Dropdown(
                    df["bedrooms"].unique(),
                    df["bedrooms"].unique(),
                    id="bedrooms-selection",
                    multi=True,
                ),
            ],
            style={"width": "48%", "display": "inline-block"},
        ),
        html.Div(
            [
                html.H3(
                    children="Wybierz typ własności nieruchomości",
                    style={"textAlign": "center", "color": colors["text"]},
                    className="card",
                ),
                dcc.Dropdown(
                    df["propertyType"].unique(),
                    df["propertyType"].unique(),
                    id="type-selection",
                    multi=True,
                ),
            ],
            style={"width": "48%", "float": "right", "display": "inline-block"},
        ),
        # html.Br(),
        html.Div(
            [
                html.H3(
                    children="Wybierz zakres lat sprzedaży nieruchomości",
                    style={"textAlign": "center", "color": colors["text"]},
                    className="card",
                ),
                dcc.RangeSlider(
                    df["Year"].min(),
                    df["Year"].max(),
                    step=None,
                    id="date-selection",
                    value=[df["Year"].min(), df["Year"].max()],
                    marks={str(year): str(year) for year in df["Year"].unique()},
                ),
            ]
        ),
        html.Div(dcc.Graph(id="chart")),
        html.Div(
            [
                html.H3(
                    children="Wybierz metodę przewidywania, aby zobaczyć jej skuteczność",
                    style={"textAlign": "center", "color": colors["text"]},
                    className="card",
                ),
                dcc.Dropdown(np.unique(methods_array), None, id="method-selection"),
            ]
        ),
        html.Div(dcc.Graph(id="chart2")),
    ],
)


@callback(
    Output("chart", "figure"),
    Output("chart2", "figure"),
    Input("bedrooms-selection", "value"),
    Input("type-selection", "value"),
    Input("date-selection", "value"),
    Input("method-selection", "value"),
)
def update_graph(
    selected_bedrooms_value: str,
    selected_type_value: str,
    dates_selection_value: str,
    selected_method_value: str,
) -> Any:
    tmp = df.loc[df.loc[:, "bedrooms"].isin(selected_bedrooms_value), :]
    tmp = tmp.loc[tmp.loc[:, "propertyType"].isin(selected_type_value), :]
    tmp = tmp[tmp.loc[:, "Year"] <= dates_selection_value[1]]
    tmp = tmp[tmp.loc[:, "Year"] >= dates_selection_value[0]]

    fig = px.line(
        tmp,
        x="datesold",
        y="price",
        labels={
            "datesold": "Data sprzedaży",
            "price": "Cena nieruchomości",
        },
    )

    if selected_method_value == "Decision tree":
        prediction_DT, rmse_DT = Decision_Tree_predict(
            os.path.join(dir, "model_DT.pkl"),
            X_test,
            Y_test,
        )
        fig2 = plot_predict(df, train_len, prediction_DT, df.price, df.datesold)
    elif selected_method_value == "Random Forest":
        prediction_RF, rmse_RF = Random_Forest_predict(
            os.path.join(dir, "model_FR.pkl"),
            X_test,
            Y_test,
        )
        fig2 = plot_predict(df, train_len, prediction_RF, df.price, df.datesold)
    else:
        fig2 = fig

    annotations = []
    annotations.append(
        dict(
            xref="paper",
            yref="paper",
            x=0.0,
            y=1.05,
            xanchor="left",
            yanchor="bottom",
            text="Ceny sprzedaży nieruchomości",
            font=dict(family="Arial", size=30, color=colors["text"]),
            showarrow=False,
        )
    )

    fig.update_layout(plot_bgcolor=colors["background"], annotations=annotations)
    return fig, fig2


if __name__ == "__main__":
    app.run_server(debug=True)