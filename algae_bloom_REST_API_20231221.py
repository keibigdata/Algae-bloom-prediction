# ------------------------------------------------------------------------
# 인공지능 녹조 예측 모델
# Copyright (c) 2023 한국공학대학교 이동현 (madeby2@gmail.com) All right reserved
# 개발일: 2023. 08. 21
# ------------------------------------------------------------------------


import pandas as pd
import numpy as np
import requests
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import csv
import os
from keras.models import Model
from keras.layers import Input, Conv1D, Dense, BatchNormalization, Dropout, GlobalAveragePooling1D
from sklearn.metrics import mean_squared_error, r2_score
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def get_data_from_openapi(start_year, end_year):
    years = [str(year) for year in range(start_year, end_year+1)]
    year_string = ','.join(years)
    
    url = 'http://apis.data.go.kr/1480523/WaterQualityService/getWaterMeasuringList'
    params = {
        'serviceKey' : '',
        'resultType' : 'JSON',
        'wmyrList' : year_string,
        'wmodList' : '01,02,03,04,05,06,07,08,09,10,11,12',
        'numOfRows' : '9999',
        'pageNo' : '1',
        'ptNoList' : '2001A60,2002A50,2003A40,2004A90,2009A10,2009A20,2012A70,2014A20,2019A80,2020A10,2022A10,2022A35,2201A48,3013A50,3101A60,4004A10'
    }
    
    try:
        response = requests.get(url, params=params, verify=False, timeout=300)
        response.encoding = 'utf-8'
        json_data = response.text 
        json_ob = json.loads(json_data) 

        if json_ob['getWaterMeasuringList']['header']['message'] != 'NORMAL SERVICE':
            print("서버 오류로 Open API 데이터가 정상적으로 다운로드 되지 않습니다.")
            return

        json_ob = json_ob['getWaterMeasuringList']['item'] 
        data = pd.json_normalize(json_ob)
        
        return data
    except requests.exceptions.Timeout:
        print("요청 시간이 300초를 초과했습니다.")
        return
    
    except Exception as e:
        print("기타 이유(요청 횟수 초과 등)로 Open API 데이터가 정상적으로 다운로드 되지 않습니다.")
        print("An error occurred:", e)
        return


def preprocess_data(data):
    mapping_dict = {
        'PT_NM': '측정소 명',
        'WMYR': '연도',
        'WMOD': '월',
        'WMWK' : '검사회차',
        'WMCYMD' : '채수 일자',
        'ITEM_TEMP': 'Water_Temperature',
        'ITEM_PH': 'pH',
        'ITEM_DOC': 'DO',
        'ITEM_BOD': 'BOD',
        'ITEM_COD': 'COD',
        'ITEM_SS': 'SS',
        'ITEM_TN': 'T-N',
        'ITEM_TP': 'T-P',
        'ITEM_CLOA': 'Chlorophyll-a',
        'ITEM_EC': 'EC',
        'ITEM_NO3N': 'NO3-N',
        'ITEM_NH3N': 'NH3-N',
        'ITEM_POP': 'PO4-P',
        'ITEM_DTN': 'DTN',
        'ITEM_DTP': 'DTP'
    }
    data = data.rename(columns=mapping_dict)
    data = data[list(mapping_dict.values())]
    
    desired_order = [
    '연도', '월', '검사회차', '채수 일자', '측정소 명',
    'BOD', 'Chlorophyll-a', 'COD', 'DO', 'DTN', 'DTP', 'EC', 'NH3-N', 'NO3-N', 'pH', 'PO4-P', 'SS', 'Water_Temperature', 'T-N', 'T-P'
    ]
    existing_columns = [col for col in desired_order if col in data.columns]
    data = data[existing_columns]
    
    station_mapping = {
        '고령': {'name': 'Goryeong', 'latitude': 35.750056, 'longitude': 128.389678},
        '곡교천2': {'name': 'Gokgyocheon2', 'latitude': 36.821639, 'longitude': 126.933944},
        '구포': {'name': 'Gupo', 'latitude': 35.216586, 'longitude': 128.995703},
        '금호강8': {'name': 'Geumhogang8', 'latitude': 35.853797, 'longitude': 128.471908},
        '남강7': {'name': 'Namgang7', 'latitude': 35.390736, 'longitude': 128.429231},
        '남지': {'name': 'Namji', 'latitude': 35.380864, 'longitude': 128.473664},
        '내성천5': {'name': 'Naeseongcheon5', 'latitude': 36.587222, 'longitude': 128.305358},
        '논산천4': {'name': 'Nonsancheon4', 'latitude': 36.165931, 'longitude': 127.010636},
        '물금': {'name': 'Mulgeum', 'latitude': 35.315297, 'longitude': 128.972189},
        '반변천4': {'name': 'Banbyeoncheon4', 'latitude': 36.551053, 'longitude': 128.751433},
        '산곡': {'name': 'Sangok', 'latitude': 36.2731, 'longitude': 128.3424},
        '상주3': {'name': 'Sangju3', 'latitude': 36.354642, 'longitude': 128.294678},
        '안동1': {'name': 'Andong1', 'latitude': 36.5809, 'longitude': 128.7644},
        '안동4': {'name': 'Andong4', 'latitude': 36.5399, 'longitude': 128.4629},
        '적성': {'name': 'Jeokseong', 'latitude': 35.400667, 'longitude': 127.220028},
        '학성': {'name': 'Hakseong', 'latitude': 35.54925, 'longitude': 129.34025}
    }
    
    def map_station_name(station):
        return station_mapping.get(station, {}).get('name', station)

    def get_latitude(station):
        return station_mapping.get(station, {}).get('latitude', None)

    def get_longitude(station):
        return station_mapping.get(station, {}).get('longitude', None)

    data['Original_측정소 명'] = data['측정소 명'].copy()
    data['측정소 명'] = data['Original_측정소 명'].apply(map_station_name)
    data['latitude'] = data['Original_측정소 명'].apply(get_latitude)
    data['longitude'] = data['Original_측정소 명'].apply(get_longitude)

    col_idx = data.columns.get_loc('측정소 명')
    data.insert(col_idx + 1, 'latitude_new', data['latitude'])
    data.insert(col_idx + 2, 'longitude_new', data['longitude'])
    data.drop(columns=['latitude', 'longitude', 'Original_측정소 명'], inplace=True)
    data.rename(columns={'latitude_new': 'latitude', 'longitude_new': 'longitude'}, inplace=True)

    # 날짜 데이터 전처리
    data['연도'] = data['연도'].astype(int)
    data['월'] = data['월'].astype(int)
    data['검사회차'] = data['검사회차'].str.replace('회차', '').astype(int)
    
    # 데이터 숫자 변환
    variables = ['BOD', 'Chlorophyll-a', 'COD', 'DO', 'DTN', 'DTP', 'EC', 'NH3-N', 'NO3-N', 'pH', 'PO4-P', 'SS', 'Water_Temperature', 'T-N', 'T-P']
    for var in variables:
        data[var] = pd.to_numeric(data[var], errors='coerce')

   # 일자별 오름차순 정렬
    data = data.sort_values(by=['측정소 명', '연도', '월', '검사회차'])
    
    # 중복 데이터 제거
    data = data.drop_duplicates()
    data = data.reset_index(drop=True)
    
    # 결측 값 전처리
    columns_to_fill = ['PO4-P', 'DTP', 'T-P', 'NH3-N']
    data[columns_to_fill] = data[columns_to_fill].fillna(0)
    data = data.groupby('측정소 명', as_index=False).apply(lambda group: group.interpolate(method='linear'))
    
    return data


def preprocess_data_by_station_for_openapi(df, time_step, lag):
    def preprocess_single_station(sub_df, station_name):
        sub_df = sub_df.copy()
        sub_df.drop(columns=['연도', '월', '검사회차', '채수 일자', '측정소 명', 'latitude', 'longitude'], inplace=True, errors='ignore')

        input_data = []
        forecast_dates = []
        station_names = []
        
        for i in range(len(sub_df) - time_step + 1):
            input_data.append(sub_df.iloc[i:i+time_step].values)
            
            # Calculate the forecast date based on the next check after 'lag' checks
            current_year = df['연도'].iloc[i + time_step - 1]
            current_month = df['월'].iloc[i + time_step - 1]
            current_check = df['검사회차'].iloc[i + time_step - 1]
                
            forecast_check = current_check + lag
            while forecast_check > 4:
                forecast_check = forecast_check - 4
                current_month += 1
                if current_month > 12:
                    current_month = 1
                    current_year += 1

            forecast_date_str = f"{current_year}.{current_month}.{forecast_check}W"
            forecast_dates.append(forecast_date_str)
            station_names.append(station_name)

        return np.array(input_data), forecast_dates, station_names
    
    stations = df['측정소 명'].unique()
    combined_X, combined_forecast_dates, combined_station_names = [], [], []

    for station in stations:
        station_data = df[df['측정소 명'] == station]
        X, forecast_dates, station_names = preprocess_single_station(station_data, station)

        combined_X.extend(X)
        combined_forecast_dates.extend(forecast_dates)
        combined_station_names.extend(station_names)

    return np.array(combined_X), combined_forecast_dates, combined_station_names


def create_model(X_train):
    n_features = X_train.shape[2]
    input = Input(shape=(None, n_features))
    x = BatchNormalization()(input)
    x = Conv1D(filters=128, kernel_size=4, activation='relu')(x)
    x = BatchNormalization()(input)
    x = Conv1D(filters=64, kernel_size=2, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Conv1D(filters=32, kernel_size=2, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    output = Dense(1)(x)
    model = Model(inputs=input, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.legacy.Nadam(), loss=tf.keras.losses.Huber())
    
    return model


def load_model_weights(X_train_shape, weights_path='./model/model_weights_v1_20230822.h5'):
    model = create_model(X_train_shape)
    model.load_weights(weights_path)
    
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    rmse_over_sd = rmse / np.std(y_test)
    print('[Performance of the Integrated Algal Bloom Prediction Model v1.0]')
    print('RMSE: {:.3f}'.format(rmse))
    print('R2: {:.3f}'.format(r2))
    print('RMSE/SD: {:.3f}'.format(rmse_over_sd))
    print('')
    
    return rmse, r2, rmse_over_sd


def get_actual_value(station_name, year, month, check, data):
    # 주어진 측정소 명, 연도, 월, 회차 조건에 맞는 실제 값을 반환
    actual_data = data[(data['측정소 명'] == station_name) & (data['연도'] == year) & (data['월'] == month) & (data['검사회차'] == check)]
    if not actual_data.empty:
        return actual_data['Chlorophyll-a'].values[0]
    else:
        return None

def save_predictions_to_csv(predictions, forecast_dates, station_names, data, filename='predictions_restapi.csv'):
    filename = os.path.join(UPLOAD_FOLDER, filename)
    actual_values = []
    
    # 실제 값을 구함
    for i, date in enumerate(forecast_dates):
        year, month, check = map(int, date.replace('W','').split('.'))
        actual_value = get_actual_value(station_names[i], year, month, check, data)
        actual_values.append(actual_value)
    
    # CSV 파일로 저장
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['측정소 명', '예측 일자', 'Prediction', 'Actual']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for i, pred in enumerate(predictions):
            writer.writerow({'측정소 명': station_names[i], '예측 일자': forecast_dates[i], 'Prediction': pred[0], 'Actual': actual_values[i]})
    
    print(f"예측 수치가 {filename}에 저장되었습니다.")
    
    return filename
    

def plot_grouped_values(station_names, forecast_dates, rounded_predictions, actual_values, filename='visualization.png'):
    filename = os.path.join(UPLOAD_FOLDER, filename)
    # 측정소 명 별로 데이터를 그룹화
    grouped_values = {}
    for station, date, pred, actual in zip(station_names, forecast_dates, rounded_predictions, actual_values):
        if station not in grouped_values:
            grouped_values[station] = {"dates": [], "predictions": [], "actuals": []}
        grouped_values[station]["dates"].append(date)
        grouped_values[station]["predictions"].append(pred[0] if isinstance(pred, (list, np.ndarray)) else pred)
        grouped_values[station]["actuals"].append(actual)

    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    fig.tight_layout(pad=5.0)

    for idx, (station, data) in enumerate(grouped_values.items()):
        row = idx // 4
        col = idx % 4

        axes[row, col].plot(data["dates"], data["predictions"], marker='o', linestyle='-', color='blue', label='Prediction')
        axes[row, col].plot(data["dates"], data["actuals"], marker='x', linestyle='-', color='red', label='Actual')
        axes[row, col].set_title(station)
        axes[row, col].set_ylabel('Chlorophyll-a')
        total_dates = len(data["dates"])
        interval = total_dates // 4
        if interval == 0:
            interval = 1
        axes[row, col].set_xticks(data["dates"][::interval])

        axes[row, col].tick_params(axis='x', rotation=45)
        axes[row, col].grid(True)
        axes[row, col].legend()
    
    plt.savefig(filename)
    plt.close()
    print(f"예측 시각화가 {filename}에 저장되었습니다.")
    
    return filename
    

def main(start_year, end_year):
    # 데이터 다운로드
    raw_data = get_data_from_openapi(start_year, end_year)

    # 데이터 전처리
    data = preprocess_data(raw_data)

    # 모델 적합 데이터 전처리
    combined_X, forecast_dates, station_names = preprocess_data_by_station_for_openapi(data, time_step=4, lag=1)

    # 모델 로드
    model = load_model_weights(combined_X)

    # 예측
    predictions = model.predict(combined_X)
    rounded_predictions = np.round(predictions, 2)

    # 예측 결과 저장
    csv_filename = save_predictions_to_csv(rounded_predictions, forecast_dates, station_names, data, filename='predictions_restapi.csv')

    # 실제값 계산 (시각화를 위해)
    actual_values = [get_actual_value(station, *map(int, date.replace('W', '').split('.')), data) for station, date in zip(station_names, forecast_dates)]

    # 시각화 및 이미지 저장
    img_filename = plot_grouped_values(station_names, forecast_dates, rounded_predictions, actual_values, filename='visualization.png')
    
    return csv_filename, img_filename

@app.route('/generate', methods=['GET'])
def generate_files():
    # 시작 연도와 끝 연도를 쿼리 파라미터로 받습니다.
    start_year = request.args.get('start_year', default=2022, type=int)
    end_year = request.args.get('end_year', default=2023, type=int)

    # main 함수에 매개변수를 전달하여 파일들을 생성합니다.
    csv_filename, img_filename = main(start_year, end_year)

    return jsonify({
        "csv_filename": csv_filename,
        "img_filename": img_filename
    })
    
    
@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, port=5000, threaded=False)