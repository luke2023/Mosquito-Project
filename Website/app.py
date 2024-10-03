import statsmodels.api as sm
import matplotlib.pyplot as plt
import io
import datetime
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
import math
import shap

# Initialize the Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Path to the CSV file
DATA_FILE = 'data/mosquito_data.csv'

# Global variable to hold mosquito data (initially empty or loaded from a CSV)
mosquito_data = pd.DataFrame()

@app.route('/admin', methods=['GET', 'POST'])
def admin():
    global mosquito_data
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Load the new data from the uploaded CSV file
            new_data = pd.read_csv(file)
            mosquito_data = new_data

            # Optionally save the updated data to a CSV file
            mosquito_data.to_csv('data/mosquito_data.csv', index=False)

            flash('Data uploaded successfully!', 'success')
            return redirect(url_for('admin'))
    return render_template('admin.html')

# Main index route for predictions
@app.route('/', methods=['GET'])
def index():
    # Get predictions and metrics for the entire dataset
    global importance_dict
    data = pd.read_csv(DATA_FILE)
    data['date'] = pd.to_datetime(data['date'], errors='coerce')  # Manually parse 'date' after reading
    predictions, metrics, importance_dict , shap = make_predictions(data)
    today = datetime.datetime.now().date()
    tomorrow = today + datetime.timedelta(days=1)

    # Date ranges for the past and future
    past_dates = [(today - datetime.timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 8)]
    future_dates = [(today + datetime.timedelta(days=i)).strftime('%Y-%m-%d') for i in range(2, 8)]
    
    # Selected date
    selected_date_str = request.args.get('selected_date', today.strftime('%Y-%m-%d'))
    # Fetch the prediction
    prediction = make_prediction_for_date(selected_date_str)
    # Fetch prediction for the selected date
    return render_template('index.html',
                           predictions=round(prediction,2),
                           metrics=metrics,
                           today=today.strftime('%Y-%m-%d'),
                           tomorrow=tomorrow.strftime('%Y-%m-%d'),
                           past_dates=past_dates,
                           future_dates=future_dates,
                           selected_date=selected_date_str,
                           feature_importances=importance_dict)

# Route to display a plot
@app.route('/plot.png')
def plot_png():
    try:
        data = pd.read_csv(DATA_FILE)
        data['date'] = pd.to_datetime(data['date'], errors='coerce')  # Manually parse 'date'
    except FileNotFoundError:
        flash('No data available to generate the plot.', 'danger')
        return redirect(url_for('index'))

    predictions,metrics, importance_dict, shap  = make_predictions(data)
    fig = create_plot(data, predictions)
    
    # Save plot to BytesIO and return as PNG
    output = io.BytesIO()
    fig.savefig(output, format='png')
    output.seek(0)
    plt.close(fig)
    return send_file(output, mimetype='image/png')
@app.route('/feature_importances.png')
def feature_importances_png():
        # Convert importance_dict back into a DataFrame
        importance_df = pd.DataFrame(importance_dict)
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        # Create the plot
        fig = plt.figure(figsize=(10, 6))
        plt.barh(importance_df['Feature'], importance_df['Importance'], color='blue')
        plt.xlabel('Importance')
        plt.title('Feature Importances from Random Forest')
        plt.gca().invert_yaxis()
        plt.tight_layout()

        # Save plot to BytesIO and return as PNG
        output = io.BytesIO()
        fig.savefig(output, format='png')
        output.seek(0)
        plt.close(fig)
        return send_file(output, mimetype='image/png')
@app.route('/shap_plot.png')
def shap_plot_png():
    # Return SHAP plot as PNG for web display
    data = pd.read_csv(DATA_FILE)
    shap_output = make_predictions(data)[3]  # Assuming make_predictions returns shap_output as the 4th item
    return send_file(shap_output, mimetype='image/png')


def make_predictions(data):
    global mean_error
    global std_error
    """Generate predictions using SARIMA and Random Forest models."""
    train_data = data[data['mosquito_density'].notna()]
    predict_data = data[data['mosquito_density'].isna()]

    if train_data.empty:
        flash('No training data available.', 'danger')
        return None, {}

    # Train SARIMA model
    mosquito_density_train = train_data['mosquito_density']
    weather_features_train = train_data[['temp', 'precip', 'solarradiation', 'humidity', 'windspeed']]
    sarima_model = sm.tsa.statespace.SARIMAX(mosquito_density_train,
                                             order=(2, 1, 2),
                                             seasonal_order=(1, 1, 1, 7),
                                             enforce_stationarity=False,
                                             enforce_invertibility=False)
    try:
        sarima_fit = sarima_model.fit(disp=False)
    except Exception as e:
        flash(f'SARIMA model fitting failed: {e}', 'danger')
        return None, {}

    # Make predictions with SARIMA
    prediction_steps = len(data)
    sarima_forecast = sarima_fit.get_prediction(start=0, end=prediction_steps - 1).predicted_mean
    sarima_forecast_train = sarima_fit.get_prediction(start=0, end=len(train_data) - 1).predicted_mean
    # Train Random Forest model on SARIMA residuals
    residuals = mosquito_density_train - sarima_forecast_train
    rf_model = RandomForestRegressor(n_estimators=500, random_state=42)
    rf_model.fit(weather_features_train, residuals)
    
    # Predict residuals for all data
    weather_features_all = data[['temp', 'precip', 'solarradiation', 'humidity', 'windspeed']]
    rf_residuals = rf_model.predict(weather_features_all)

    # Combine SARIMA forecast and RF residuals for hybrid prediction
    hybrid_forecast = sarima_forecast + rf_residuals
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(weather_features_train)

    # Save SHAP summary plot
    shap_fig = plt.figure()
    shap.summary_plot(shap_values, weather_features_train, show=False)
    shap_output = io.BytesIO()
    shap_fig.savefig(shap_output, format='png')
    shap_output.seek(0)
    plt.close(shap_fig)
    # Save predictions
    predictions_df = pd.DataFrame({
        'date': data['date'],
        'predicted_mosquito_density': hybrid_forecast
    })
    predictions_file = 'predicted_mosquito_density.csv'
    predictions_df.to_csv(predictions_file, index=False)
    # Calculate metrics
    mae = mean_absolute_error(mosquito_density_train, sarima_forecast_train + rf_model.predict(weather_features_train))
    rmse = np.sqrt(mean_squared_error(mosquito_density_train, sarima_forecast_train + rf_model.predict(weather_features_train)))
    mean_error = np.mean((-mosquito_density_train + (sarima_forecast_train + rf_model.predict(weather_features_train))) / (sarima_forecast_train + rf_model.predict(weather_features_train)))
    std_error = np.std((-mosquito_density_train + (sarima_forecast_train + rf_model.predict(weather_features_train))) / (sarima_forecast_train + rf_model.predict(weather_features_train)))
    importances = rf_model.feature_importances_
    features = weather_features_train.columns
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Convert the DataFrame to a dictionary for passing to the template
    importance_dict = importance_df.to_dict(orient='records')
    # Package the metrics
    metrics = {
        'MAE': round(mae,2),
        'RMSE': round(rmse,2),
        'Mean Error': round(mean_error*100,2),
        'STD (Percentage)': round(std_error * 100,2)  # Convert to percentage
    }

    return hybrid_forecast.tolist(), metrics , importance_dict,shap_output


def make_prediction_for_date(selected_date_str):
    """Fetch prediction for a specific date from the predictions file."""
    # Load the predictions CSV file
    predictions_df = pd.read_csv('predicted_mosquito_density.csv')

    # Ensure the 'date' column is in string format to match with selected_date_str
    predictions_df['date'] = pd.to_datetime(predictions_df['date']).dt.strftime('%Y-%m-%d')

    # Fetch the prediction for the selected date
    prediction = predictions_df[predictions_df['date'] == selected_date_str]['predicted_mosquito_density']
    
    # Check if any prediction exists for the selected date
    if prediction.empty:
        return f"No prediction found for the date: {selected_date_str}"
    
    # Return the first prediction found for the selected date
    return prediction.values[0]

    
def create_plot(data, predictions):
    """Create a plot of actual and predicted mosquito density."""
    errors_up = np.abs((mean_error + std_error) * np.array(predictions))
    errors_down = np.abs((std_error - mean_error) * np.array(predictions))
    errors = [errors_down, errors_up]

    data_sorted = data.sort_values('date')
    data_sorted['date'] = pd.to_datetime(data_sorted['date'])
    data_sorted['predicted_density'] = predictions
    date_range = pd.date_range(start='2024-09-05', periods=len(predictions), freq='D')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(date_range, data_sorted['mosquito_density'], label='Actual Mosquito Density', color='red', marker='o')
    ax.errorbar(date_range, data_sorted['predicted_density'], yerr=errors, label='Predicted Mosquito Density', color='purple', marker='x', capsize=3)
    ax.set_title('Mosquito Density Predictions')
    ax.set_xlabel('Date')
    ax.set_ylabel('Mosquito Density')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig

if __name__ == '__main__':
    app.run(debug=True)






