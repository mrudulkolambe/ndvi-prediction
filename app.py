from flask import Flask, jsonify, render_template
import requests
from datetime import datetime, timedelta
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

app = Flask(__name__)

# @app.route("/", methods=["GET"])
# def mbsa():
#     return render_template('index.html')

@app.route('/api', methods=['GET'])
def get_data():
    start_timestamp = int((datetime.now() - timedelta(days=3*30)).timestamp())
    end_timestamp = int(datetime.now().timestamp())
    api_url = f"https://api.agromonitoring.com/agro/1.0/ndvi/history?polyid=662de9cd1a451ad923bed17a&start={start_timestamp}&end={end_timestamp}&appid=f742fa923a192f5179a6f04ed62092cd"
    response = requests.get(api_url)

    if response.status_code == 200:
        data = response.json()
        data1 = [float(item['data']['max']) for item in data]
        data2 = [item['dt'] for item in data]
        df = pd.DataFrame({'data__max': data1, 'dt': data2})
        df['dt'] = pd.to_datetime(df['dt'])

        X = df[['dt']]
        y = df['data__max']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

        rf_regressor.fit(X_train, y_train)

        y_pred = rf_regressor.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)

        future_timestamps = pd.date_range(start='2024-04-29', end='2024-04-29', freq='D')
        future_X = pd.DataFrame({'dt': future_timestamps})
        future_predictions = rf_regressor.predict(future_X)


        # Example parameters
        current_ndvi = future_predictions[0]
        desired_ndvi = 1.0
        area_size_sqm = 279/3
        canopy_size_sqm = 4
        spacing_m = 2

        # num_trees_needed = calculate_trees_needed_for_ndvi(current_ndvi, desired_ndvi, area_size_sqm, canopy_size_sqm, spacing_m)
        delta_ndvi = desired_ndvi - current_ndvi
        vegetation_increase = delta_ndvi / current_ndvi
        tree_coverage_sqm = canopy_size_sqm * 0.8  
        area_covered_needed = area_size_sqm * vegetation_increase
        num_trees = area_covered_needed / tree_coverage_sqm
        num_trees /= (spacing_m ** 2)
        # print("Number of trees needed to increase NDVI to", desired_ndvi, ":", r)
        return jsonify({
            "trees": round(num_trees),
            "error": mse,
            "ndvi": current_ndvi
        })




def calculate_trees_needed_for_ndvi(current_ndvi, desired_ndvi, area_size_sqm, canopy_size_sqm, spacing_m):
    delta_ndvi = desired_ndvi - current_ndvi
    vegetation_increase = delta_ndvi / current_ndvi
    tree_coverage_sqm = canopy_size_sqm * 0.8  
    area_covered_needed = area_size_sqm * vegetation_increase
    num_trees = area_covered_needed / tree_coverage_sqm
    num_trees /= (spacing_m ** 2)
    return round(num_trees)



if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)  # Run the app in debug mode