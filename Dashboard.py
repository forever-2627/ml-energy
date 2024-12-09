import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import google.generativeai as genai
from datetime import datetime
from openmeteopy import OpenMeteo
from openmeteopy.hourly import HourlyForecast
from openmeteopy.options import ForecastOptions
import os
import holidays
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Google API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

project_path = os.getcwd()
# Set page configuration for full-width layout
st.set_page_config(layout="wide", page_title="Energy Usage Dashboard")

# Load datasets
@st.cache_data
def load_main_data():
    df = pd.read_parquet(f"{project_path}/energy_data.parquet")
    # Convert 'date' to datetime if needed
    # If 'date' is already in datetime format, this step is not required
    if df['date'].dtype != 'datetime64[ns]':
        df['date'] = pd.to_datetime(df['date'])

    # Extract month name and hour from the 'date' column
    df["month"] = df["date"].dt.month_name()
    df["hour"] = df["date"].dt.hour

    return df

@st.cache_data
def load_llm_data():
    df = pd.read_parquet(f"{project_path}/daily_data.parquet")
    df['date'] = pd.to_datetime(df['date'])
    df['hour'] = df['hour'].astype(int)
    return df[df['outlier'] == 0]

# Create LLM context
def create_data_context(df):
    context = "Energy consumption analysis by appliance:\n\n"
    appliance_avg = df.groupby('appliance')['usage'].mean()
    context += "Average usage by appliance (kWh):\n"
    for appliance, usage in appliance_avg.items():
        context += f"{appliance}: {usage:.4f}\n"
    context += "\nSeasonal patterns:\n"
    for season in df['season'].unique():
        season_data = df[df['season'] == season]
        context += f"\n{season}:\n"
        for appliance in df['appliance'].unique():
            avg_usage = season_data[season_data['appliance'] == appliance]['usage'].mean()
            if not pd.isna(avg_usage):
                context += f"{appliance}: {avg_usage:.4f} kWh\n"
    return context

def query_model(question, context):
    prompt = f"""Based on this appliance-specific energy consumption data:

    {context}

    Question: {question}
    
    Provide a clear, data-backed answer focusing on the specific appliances mentioned."""
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text

# Create visualization for LLM tab
def create_plot(df, plot_type, selected_appliances=None):
    if selected_appliances is None or len(selected_appliances) == 0:
        return None
    fig, ax = plt.subplots(figsize=(12, 6))
    if plot_type == "hourly":
        for appliance in selected_appliances:
            hourly_avg = df[df['appliance'] == appliance].groupby('hour')['usage'].mean()
            ax.plot(hourly_avg.index, hourly_avg.values, marker='o', label=appliance)
        ax.set_title('Average Energy Usage by Hour')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Energy Usage (kWh)')
        ax.legend()
    elif plot_type == "appliance_comparison":
        avg_usage = df[df['appliance'].isin(selected_appliances)].groupby('appliance')['usage'].mean()
        ax.bar(avg_usage.index, avg_usage.values)
        ax.set_title('Average Usage by Appliance')
        ax.set_xlabel('Appliance')
        ax.set_ylabel('Usage (kWh)')
    elif plot_type == "seasonal":
        seasonal_data = df[df['appliance'].isin(selected_appliances)].pivot_table(
            values='usage', index='season', columns='appliance', aggfunc='mean'
        )
        seasonal_data.plot(kind='bar', ax=ax)
        ax.set_title('Seasonal Usage by Appliance')
        ax.set_xlabel('Season')
        ax.set_ylabel('Usage (kWh)')
    plt.tight_layout()
    return fig

def load_content(file_name):
    with open(file_name, "r") as file:
        return file.read()

def get_model_input(datetime_list):
    us_holidays = holidays.US()
    
    # Prepare a list to store each row of the DataFrame
    data = []
    
    for datetime in datetime_list:
        datetime = pd.Timestamp(datetime)
        month = datetime.month
        day_name = datetime.day_name()
        hour = datetime.hour
        
        # Check if it's a weekend or holiday
        is_weekend = datetime.weekday() >= 5  # Saturday is 5, Sunday is 6
        is_holiday = (datetime in us_holidays) or is_weekend
        season = ''
        if month in [12, 1, 2]:
            season = 'Winter'
        elif month in [3, 4, 5]:
            season = 'Spring'
        elif month in [6, 7, 8]:
            season = 'Summer'
        elif month in [9, 10, 11]:
            season = 'Fall'

        # Append each row as a dictionary
        data.append({
            'month': month,
            'day_name': day_name,
            'hour': hour,
            'is_holiday': is_holiday,
            'season': season
        })
    
    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(data)
    return df
def get_predict_dataset(date_data, weather_data):
    input_subset = date_data[['month', 'day_name', 'hour', 'season', 'is_holiday']]

    # Extract specific columns from weather DataFrame
    weather_subset = weather_data[['temperature_2m', 'relativehumidity_2m', 'windspeed_10m']]

    # Rename weather columns to match your desired keys
    weather_subset = weather_subset.rename(columns={
        'temperature_2m': 'temp',
        'relativehumidity_2m': 'rhum',
        'windspeed_10m': 'wspd'
    })

    # Concatenate the extracted information into a new DataFrame
    combined_df = pd.concat([input_subset, weather_subset], axis=1)

    # Display the combined DataFrame

    categorical_cols = combined_df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = combined_df.select_dtypes(include=['number']).columns.tolist()
    for col in categorical_cols:
        le = LabelEncoder()
        combined_df[col] = le.fit_transform(combined_df[col])

    # Update preprocessor to only scale numerical columns
    preprocessor_num = Pipeline(steps=[('scale', StandardScaler())])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', preprocessor_num, numerical_cols)
        ],
        remainder='passthrough'  # Keep categorical columns encoded as integers
    )

    # Apply the updated transformations
    X_preprocessed = preprocessor.fit_transform(combined_df)
    return X_preprocessed

def get_weather_data(longitude, latitude):
    hourly = HourlyForecast()
    options = ForecastOptions(latitude, longitude)

    # Create OpenMeteo objects for each type of data
    temperature = OpenMeteo(options, hourly.temperature_2m())
    humidity = OpenMeteo(options, hourly.relativehumidity_2m())
    pressure = OpenMeteo(options, hourly.surface_pressure())
    windspeed = OpenMeteo(options, hourly.windspeed_10m())

    # Download data into separate DataFrames
    data = temperature.get_pandas()
    data = humidity.get_pandas()
    data = pressure.get_pandas()
    data = windspeed.get_pandas()

    # Ensure the 'time' column is set if not
    if 'time' not in data.columns:
        data.reset_index(inplace=True)  # If the time is in the index rather than a column

    return data
# Load your CSS
custom_style = load_content("custom.css")

st.markdown(f"<style>{custom_style}</style>", unsafe_allow_html=True)

# Load data
main_df = load_main_data()
llm_df = load_llm_data()

# Sidebar filters
st.sidebar.title("Filters")
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(main_df["date"].min().date(), main_df["date"].max().date()),
    min_value=main_df["date"].min().date(),
    max_value=main_df["date"].max().date()
)
city_filter = st.sidebar.multiselect("City", options=main_df["city"].unique(), default=main_df["city"].unique())
season_filter = st.sidebar.multiselect("Season", options=main_df["season"].unique(), default=main_df["season"].unique())

# Apply filters
filtered_df = main_df[
    (main_df["date"].dt.date >= date_range[0]) & 
    (main_df["date"].dt.date <= date_range[1]) & 
    (main_df["city"].isin(city_filter)) & 
    (main_df["season"].isin(season_filter))
]
st.header("Energy Consumption Optimize")
# Tabs
selected_tab = option_menu(
    menu_title=None,
    options=["Dashboard", "Overview and Key Insights", "Energy Usage Trends", "Outliers Analysis", "Chat with an Expert"],
    icons=["laptop", "bar-chart", "line-chart", "pie-chart", "robot"],
    default_index=0,
    orientation="horizontal",
    key="option-menu"
)

longitude = -97.7431  # Austin Longitude and Latitude
latitude =  30.2672
weather = get_weather_data(longitude, latitude)
inputdata = get_model_input(weather['time'])

air_comp_model = joblib.load(f'{project_path}/models/aircomp.joblib')
bathroom_model = joblib.load(f'{project_path}/models/bathroom.joblib')
bedroom_model = joblib.load(f'{project_path}/models/bedroom.joblib')
car_model = joblib.load(f'{project_path}/models/car.joblib')
diningroom_model = joblib.load(f'{project_path}/models/diningroom.joblib')
grid_model = joblib.load(f'{project_path}/models/grid.joblib')
kitchen_area_model = joblib.load(f'{project_path}/models/kitchen_area.joblib')
livingroom_model = joblib.load(f'{project_path}/models/livingroom.joblib')
office_model = joblib.load(f'{project_path}/models/office.joblib')
other_model = joblib.load(f'{project_path}/models/other.joblib')
utilityroom_model = joblib.load(f'{project_path}/models/utilityroom.joblib')
washer_dryer_model = joblib.load(f'{project_path}/models/washer_dryer.joblib')
waterheater_model = joblib.load(f'{project_path}/models/waterheater.joblib')

predict_data = get_predict_dataset(inputdata, weather);

data_dict = {
    'time': weather['time'],
    'air_comp': air_comp_model.predict(predict_data),
    'bathroom': bathroom_model.predict(predict_data),
    'bedroom': bedroom_model.predict(predict_data),
    'car': car_model.predict(predict_data),
    'diningroom': diningroom_model.predict(predict_data),
    'grid': grid_model.predict(predict_data),
    'kitchen_area': kitchen_area_model.predict(predict_data),
    'livingroom': livingroom_model.predict(predict_data),
    'office': office_model.predict(predict_data),
    'other': other_model.predict(predict_data),
    'utilityroom': utilityroom_model.predict(predict_data),
    'washer_dryer': washer_dryer_model.predict(predict_data),
    'waterheater': waterheater_model.predict(predict_data)
}

predictions_df = pd.DataFrame(data_dict)

print(predictions_df.head())

# Main Interface

if selected_tab == "Dashboard":
    with st.container(key="key-metrics-container"):
        st.markdown("### **_Weather (Austin)_**")
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        
        with kpi_col1:
            temperature = 15
            st.metric("Current Temperature", f"{round(weather['temperature_2m'][0], 2)} °C")
        
        with kpi_col2:
            humidity = 15
            st.metric("Current Relative Humidity", f"{round(weather['relativehumidity_2m'][0])} %")
        
        with kpi_col3:
            pressure = 15
            st.metric("Pressure", f"{round(weather['surface_pressure'][0])}hPa")
        
        with kpi_col4:
            windspeed = 15
            st.metric("Wind Speed", f"{round(weather['windspeed_10m'][0])}m/s")

    with st.container(key="key-daily_average-graph"):
        st.markdown("### **_Weather Forecast_**")
        with st.container(key="key-temerature-graph"):
            fig_usage = px.line(
                weather,
                x="time",
                y="temperature_2m",
                title="Temperature Forecast",
                labels={"temperature_2m": "Temperature (°C)", "date": "Date"},
                template="plotly_dark"  # Optional: Use a dark template for aesthetic
            )

            # Update figure layout for aesthetics
            fig_usage.update_traces(
                line=dict(shape='spline', smoothing=1.3, width=2.5),  # Smoothen and widen line
                mode='lines',  # Add markers to each data point
                marker=dict(size=8, color='rgba(200, 50, 50, .8)')  # Marker customization
            )

            fig_usage.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="Date"),
                yaxis=dict(title="Temperature (°C)"),
                title=dict(font=dict(size=24)),  # Center the title
                margin=dict(l=40, r=40, t=40, b=40)  # Adjust margins
            )

            # Display the Plotly figure in Streamlit
            st.plotly_chart(fig_usage, use_container_width=True)
        with st.container(key="key-humidity-graph"):
            fig_usage = px.line(
                weather,
                x="time",
                y="relativehumidity_2m",
                title="Humidity Forecast",
                labels={"relativehumidity_2m": "Humidity (%)", "date": "Date"},
                template="plotly_dark"  # Optional: Use a dark template for aesthetic
            )

            # Update figure layout for aesthetics
            fig_usage.update_traces(
                line=dict(shape='spline', smoothing=1.3, width=2.5),  # Smoothen and widen line
                mode='lines',  # Add markers to each data point
                marker=dict(size=8, color='rgba(200, 50, 50, .8)')  # Marker customization
            )

            fig_usage.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="Date"),
                yaxis=dict(title="Humidity (%)"),
                title=dict(font=dict(size=24)),  # Center the title
                margin=dict(l=40, r=40, t=40, b=40)  # Adjust margins
            )

            # Display the Plotly figure in Streamlit
            st.plotly_chart(fig_usage, use_container_width=True)
        with st.container(key="key-windspeed-graph"):
            fig_usage = px.line(
                weather,
                x="time",
                y="windspeed_10m",
                title="Wind Speed Forecast",
                labels={"windspeed_10m": "Wind Speed (m/s)", "date": "Date"},
                template="plotly_dark"  # Optional: Use a dark template for aesthetic
            )

            # Update figure layout for aesthetics
            fig_usage.update_traces(
                line=dict(shape='spline', smoothing=1.3, width=2.5),  # Smoothen and widen line
                mode='lines',  # Add markers to each data point
                marker=dict(size=8, color='rgba(200, 50, 50, .8)')  # Marker customization
            )

            fig_usage.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="Date"),
                yaxis=dict(title="Wind Speed (m/s)"),
                title=dict(font=dict(size=24)),  # Center the title
                margin=dict(l=40, r=40, t=40, b=40)  # Adjust margins
            )

            # Display the Plotly figure in Streamlit
            st.plotly_chart(fig_usage, use_container_width=True)
        with st.container(key="key-aircomp-graph"):
            fig_usage = px.line(
                predictions_df,
                x="time",
                y="air_comp",
                title="Air Conditions Energy Consumption Prediction",
                labels={"air_comp": "Air Conditions Energy", "date": "Date"},
                template="plotly_dark"  # Optional: Use a dark template for aesthetic
            )

            # Update figure layout for aesthetics
            fig_usage.update_traces(
                line=dict(shape='spline', smoothing=1.3, width=2.5),  # Smoothen and widen line
                mode='lines',  # Add markers to each data point
                marker=dict(size=8, color='rgba(200, 50, 50, .8)')  # Marker customization
            )

            fig_usage.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="Date"),
                yaxis=dict(title="Air Conditions Energy"),
                title=dict(font=dict(size=24)),  # Center the title
                margin=dict(l=40, r=40, t=40, b=40)  # Adjust margins
            )

            # Display the Plotly figure in Streamlit
            st.plotly_chart(fig_usage, use_container_width=True)

        with st.container(key="key-bathroom-graph"):
            fig_usage = px.line(
                predictions_df,
                x="time",
                y="bathroom",
                title="Bathroom Energy Consumption Prediction",
                labels={"bathroom": "Bathroom Energy", "date": "Date"},
                template="plotly_dark"  # Optional: Use a dark template for aesthetic
            )

            # Update figure layout for aesthetics
            fig_usage.update_traces(
                line=dict(shape='spline', smoothing=1.3, width=2.5),  # Smoothen and widen line
                mode='lines',  # Add markers to each data point
                marker=dict(size=8, color='rgba(200, 50, 50, .8)')  # Marker customization
            )

            fig_usage.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="Date"),
                yaxis=dict(title="Bathroom Energy"),
                title=dict(font=dict(size=24)),  # Center the title
                margin=dict(l=40, r=40, t=40, b=40)  # Adjust margins
            )

            # Display the Plotly figure in Streamlit
            st.plotly_chart(fig_usage, use_container_width=True)

        with st.container(key="key-bedroom-graph"):
            fig_usage = px.line(
                predictions_df,
                x="time",
                y="bedroom",
                title="Bedroom Energy Consumption Prediction",
                labels={"bedroom": "Bedroom Energy", "date": "Date"},
                template="plotly_dark"  # Optional: Use a dark template for aesthetic
            )

            # Update figure layout for aesthetics
            fig_usage.update_traces(
                line=dict(shape='spline', smoothing=1.3, width=2.5),  # Smoothen and widen line
                mode='lines',  # Add markers to each data point
                marker=dict(size=8, color='rgba(200, 50, 50, .8)')  # Marker customization
            )

            fig_usage.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="Date"),
                yaxis=dict(title="bedroom Energy"),
                title=dict(font=dict(size=24)),  # Center the title
                margin=dict(l=40, r=40, t=40, b=40)  # Adjust margins
            )

            # Display the Plotly figure in Streamlit
            st.plotly_chart(fig_usage, use_container_width=True)

        with st.container(key="key-car-graph"):
            fig_usage = px.line(
                predictions_df,
                x="time",
                y="car",
                title="Car Energy Consumption Prediction",
                labels={"car": "Car Energy", "date": "Date"},
                template="plotly_dark"  # Optional: Use a dark template for aesthetic
            )

            # Update figure layout for aesthetics
            fig_usage.update_traces(
                line=dict(shape='spline', smoothing=1.3, width=2.5),  # Smoothen and widen line
                mode='lines',  # Add markers to each data point
                marker=dict(size=8, color='rgba(200, 50, 50, .8)')  # Marker customization
            )

            fig_usage.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="Date"),
                yaxis=dict(title="Car Energy"),
                title=dict(font=dict(size=24)),  # Center the title
                margin=dict(l=40, r=40, t=40, b=40)  # Adjust margins
            )

            # Display the Plotly figure in Streamlit
            st.plotly_chart(fig_usage, use_container_width=True)

        with st.container(key="key-diningroom-graph"):
            fig_usage = px.line(
                predictions_df,
                x="time",
                y="diningroom",
                title="Diningroom Energy Consumption Prediction",
                labels={"diningroom": "Diningroom Energy", "date": "Date"},
                template="plotly_dark"  # Optional: Use a dark template for aesthetic
            )

            # Update figure layout for aesthetics
            fig_usage.update_traces(
                line=dict(shape='spline', smoothing=1.3, width=2.5),  # Smoothen and widen line
                mode='lines',  # Add markers to each data point
                marker=dict(size=8, color='rgba(200, 50, 50, .8)')  # Marker customization
            )

            fig_usage.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="Date"),
                yaxis=dict(title="Diningroom Energy"),
                title=dict(font=dict(size=24)),  # Center the title
                margin=dict(l=40, r=40, t=40, b=40)  # Adjust margins
            )

            # Display the Plotly figure in Streamlit
            st.plotly_chart(fig_usage, use_container_width=True)

        with st.container(key="key-grid-graph"):
            fig_usage = px.line(
                predictions_df,
                x="time",
                y="grid",
                title="Grid Energy Consumption Prediction",
                labels={"grid": "Grid Energy", "date": "Date"},
                template="plotly_dark"  # Optional: Use a dark template for aesthetic
            )

            # Update figure layout for aesthetics
            fig_usage.update_traces(
                line=dict(shape='spline', smoothing=1.3, width=2.5),  # Smoothen and widen line
                mode='lines',  # Add markers to each data point
                marker=dict(size=8, color='rgba(200, 50, 50, .8)')  # Marker customization
            )

            fig_usage.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="Date"),
                yaxis=dict(title="Grid Energy"),
                title=dict(font=dict(size=24)),  # Center the title
                margin=dict(l=40, r=40, t=40, b=40)  # Adjust margins
            )

            # Display the Plotly figure in Streamlit
            st.plotly_chart(fig_usage, use_container_width=True)

        with st.container(key="key-livingroom-graph"):
            fig_usage = px.line(
                predictions_df,
                x="time",
                y="livingroom",
                title="Living Room Energy Consumption Prediction",
                labels={"livingroom": "Living Room Energy", "date": "Date"},
                template="plotly_dark"  # Optional: Use a dark template for aesthetic
            )

            # Update figure layout for aesthetics
            fig_usage.update_traces(
                line=dict(shape='spline', smoothing=1.3, width=2.5),  # Smoothen and widen line
                mode='lines',  # Add markers to each data point
                marker=dict(size=8, color='rgba(200, 50, 50, .8)')  # Marker customization
            )

            fig_usage.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="Date"),
                yaxis=dict(title="Living Room Energy"),
                title=dict(font=dict(size=24)),  # Center the title
                margin=dict(l=40, r=40, t=40, b=40)  # Adjust margins
            )

            # Display the Plotly figure in Streamlit
            st.plotly_chart(fig_usage, use_container_width=True)

        with st.container(key="key-office-graph"):
            fig_usage = px.line(
                predictions_df,
                x="time",
                y="office",
                title="Office Energy Consumption Prediction",
                labels={"office": "Office Energy", "date": "Date"},
                template="plotly_dark"  # Optional: Use a dark template for aesthetic
            )

            # Update figure layout for aesthetics
            fig_usage.update_traces(
                line=dict(shape='spline', smoothing=1.3, width=2.5),  # Smoothen and widen line
                mode='lines',  # Add markers to each data point
                marker=dict(size=8, color='rgba(200, 50, 50, .8)')  # Marker customization
            )

            fig_usage.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="Date"),
                yaxis=dict(title="Office Energy"),
                title=dict(font=dict(size=24)),  # Center the title
                margin=dict(l=40, r=40, t=40, b=40)  # Adjust margins
            )

            # Display the Plotly figure in Streamlit
            st.plotly_chart(fig_usage, use_container_width=True)

        with st.container(key="key-other-graph"):
            fig_usage = px.line(
                predictions_df,
                x="time",
                y="other",
                title="Other Energy Consumption Prediction",
                labels={"other": "Other Energy", "date": "Date"},
                template="plotly_dark"  # Optional: Use a dark template for aesthetic
            )

            # Update figure layout for aesthetics
            fig_usage.update_traces(
                line=dict(shape='spline', smoothing=1.3, width=2.5),  # Smoothen and widen line
                mode='lines',  # Add markers to each data point
                marker=dict(size=8, color='rgba(200, 50, 50, .8)')  # Marker customization
            )

            fig_usage.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="Date"),
                yaxis=dict(title="Other Energy"),
                title=dict(font=dict(size=24)),  # Center the title
                margin=dict(l=40, r=40, t=40, b=40)  # Adjust margins
            )

            # Display the Plotly figure in Streamlit
            st.plotly_chart(fig_usage, use_container_width=True)

        with st.container(key="key-utilityroom-graph"):
            fig_usage = px.line(
                predictions_df,
                x="time",
                y="utilityroom",
                title="Utility Room Energy Consumption Prediction",
                labels={"utilityroom": "Utility Room Energy", "date": "Date"},
                template="plotly_dark"  # Optional: Use a dark template for aesthetic
            )

            # Update figure layout for aesthetics
            fig_usage.update_traces(
                line=dict(shape='spline', smoothing=1.3, width=2.5),  # Smoothen and widen line
                mode='lines',  # Add markers to each data point
                marker=dict(size=8, color='rgba(200, 50, 50, .8)')  # Marker customization
            )

            fig_usage.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="Date"),
                yaxis=dict(title="Utility Room Energy"),
                title=dict(font=dict(size=24)),  # Center the title
                margin=dict(l=40, r=40, t=40, b=40)  # Adjust margins
            )

            # Display the Plotly figure in Streamlit
            st.plotly_chart(fig_usage, use_container_width=True)

        with st.container(key="key-washer_dryer-graph"):
            fig_usage = px.line(
                predictions_df,
                x="time",
                y="washer_dryer",
                title="Washer Dryer Energy Consumption Prediction",
                labels={"washer_dryer": "Washer Dryer Energy", "date": "Date"},
                template="plotly_dark"  # Optional: Use a dark template for aesthetic
            )

            # Update figure layout for aesthetics
            fig_usage.update_traces(
                line=dict(shape='spline', smoothing=1.3, width=2.5),  # Smoothen and widen line
                mode='lines',  # Add markers to each data point
                marker=dict(size=8, color='rgba(200, 50, 50, .8)')  # Marker customization
            )

            fig_usage.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="Date"),
                yaxis=dict(title="Washer Dryer Energy"),
                title=dict(font=dict(size=24)),  # Center the title
                margin=dict(l=40, r=40, t=40, b=40)  # Adjust margins
            )

            # Display the Plotly figure in Streamlit
            st.plotly_chart(fig_usage, use_container_width=True)
        
        with st.container(key="key-waterheater-graph"):
            fig_usage = px.line(
                predictions_df,
                x="time",
                y="waterheater",
                title="Water Heater Energy Consumption Prediction",
                labels={"waterheater": "Water Heater Energy", "date": "Date"},
                template="plotly_dark"  # Optional: Use a dark template for aesthetic
            )

            # Update figure layout for aesthetics
            fig_usage.update_traces(
                line=dict(shape='spline', smoothing=1.3, width=2.5),  # Smoothen and widen line
                mode='lines',  # Add markers to each data point
                marker=dict(size=8, color='rgba(200, 50, 50, .8)')  # Marker customization
            )

            fig_usage.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="Date"),
                yaxis=dict(title="Water Heater Energy"),
                title=dict(font=dict(size=24)),  # Center the title
                margin=dict(l=40, r=40, t=40, b=40)  # Adjust margins
            )

            # Display the Plotly figure in Streamlit
            st.plotly_chart(fig_usage, use_container_width=True)
elif selected_tab == "Overview and Key Insights":

    with st.container(key="key-metrics-container"):
        st.markdown("### **_Key Metrics_**")
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        
        with kpi_col1:
            avg_daily_consumption = filtered_df[filtered_df["appliance"] == "grid"].groupby("day")["usage"].mean().mean()
            st.metric("Avg. Daily Consumption (Grid)", f"{avg_daily_consumption:.2f} kWh")
        
        with kpi_col2:
            num_appliances = filtered_df["appliance"].nunique()
            st.metric("Number of Appliances", num_appliances)
        
        with kpi_col3:
            avg_outliers = filtered_df["outlier"].mean()
            st.metric("Avg. Outliers Count", f"{avg_outliers:.2f}")
        
        with kpi_col4:
            total_data_points = len(filtered_df)
            st.metric("Data Points", total_data_points)

    with st.container(key="monthly-container"):
    # Second Row Charts
        st.markdown("### **_Monthly and Appliance Insights_**")
        row2_col1, row2_col2, row2_col3 = st.columns([1, 1, 1])

        # Chart 1: Total Grid Usage (This Month vs Last Month)
        with row2_col1:
            current_month = pd.Timestamp.now().month
            this_month = filtered_df[filtered_df["date"].dt.month == current_month]
            last_month = filtered_df[filtered_df["date"].dt.month == (current_month - 1)]
            
            # Extract day of month for common X-axis
            this_month["day_of_month"] = this_month["date"].dt.day
            last_month["day_of_month"] = last_month["date"].dt.day

            # Combine this month and last month data
            this_month["label"] = "This Month"
            last_month["label"] = "Last Month"
            combined_data = pd.concat([this_month, last_month])
            combined_data = combined_data.groupby(["day_of_month", "label"])["usage"].sum().reset_index()

            fig_usage = px.line(
                combined_data,
                x="day_of_month",
                y="usage",
                color="label",
                title="Grid Usage: This Month vs Last Month",
                labels={"day_of_month": "Day of Month", "usage": "Total Usage (kWh)", "label": "Month"},
            )
            fig_usage.update_layout(
                xaxis=dict(title="Day of Month"),
                yaxis=dict(title="Total Usage (kWh)"),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_usage, use_container_width=True)
    
        # Chart 2: Top 5 Appliances by Total Monthly Usage
        with row2_col2:
            appliance_usage = filtered_df[filtered_df["appliance"] != "grid"].groupby("appliance")["usage"].sum().reset_index()
            top_appliances = appliance_usage.sort_values(by="usage", ascending=False).head(5)
            fig_appliances = px.bar(
                top_appliances,
                x="usage",
                y="appliance",
                orientation="h",
                title="Top 5 Appliances (Monthly Avg Usage)",
                labels={"appliance": "Appliance", "usage": "Total Usage (kWh)"},
            )
            fig_appliances.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_appliances, use_container_width=True)

        # Chart 3: Weekend Effect on Grid Consumption
        with row2_col3:
            weekend_usage = filtered_df.groupby("is_weekend")["usage"].mean().reset_index()
            fig_weekend = px.bar(
                weekend_usage,
                x="is_weekend",
                y="usage",
                title="Weekend Effect on Grid Consumption",
                labels={"is_weekend": "Weekend", "usage": "Daily Avg Usage (kWh)"},
            )
            fig_weekend.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_weekend, use_container_width=True)

    with st.container(key="city-container"):
    # Third and Fourth Row Chart
        st.markdown("### **_Total Grid Consumption by City (Smoothed)_**")
        # Group by city and day, then calculate daily average usage
        filtered_df["day"] = filtered_df["date"].dt.date  # Extract day (YYYY-MM-DD)
        daily_city_usage = (
            filtered_df[filtered_df["appliance"] == "grid"]
            .groupby(["day", "city"])["usage"]
            .mean()
            .reset_index()
        )

        fig_city_usage = px.line(
            daily_city_usage,
            x="day",
            y="usage",
            color="city",
            title="Total Grid Consumption by City (Daily Average)",
            labels={"day": "Date", "usage": "Daily Avg Usage (kWh)", "city": "City"},
        )
        fig_city_usage.update_layout(
            xaxis=dict(title="Date"),
            yaxis=dict(title="Daily Avg Usage (kWh)"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_city_usage, use_container_width=True)


elif selected_tab == "Energy Usage Trends":
    st.sidebar.markdown("### Filters for Energy Usage Trends")
    
    # Appliance Filter for Heatmap
    appliances_to_display = st.sidebar.multiselect(
        "Select Appliances to Display in Heatmap",
        options=filtered_df["appliance"].unique(),
        default=filtered_df["appliance"].unique(),
        help="Filter appliances to display in the heatmap."
    )

    # Selector for Heatmap X-axis
    x_axis_selector = st.sidebar.selectbox(
        "Select X-Axis for Heatmap",
        options=["Hour", "Time of Day", "Season"],
        index=0,
        help="Change the X-axis of the heatmap."
    )

    # Appliance Selector for Hourly Usage Trends
    appliances_for_hourly_plot = st.sidebar.multiselect(
        "Select Appliances for Hourly Usage Trends",
        options=filtered_df["appliance"].unique(),
        default=["grid"],  # Default to grid appliance
        help="Select appliances to display in hourly usage trends (subplots)."
    )
    with st.container(key="hourly-container"):

    # Hourly Usage Line Plot (Subplots by Appliance)
        st.markdown("### **_Hourly Usage Trends_**")
        
        # Filter data for the selected appliances
        hourly_filtered_df = filtered_df[filtered_df["appliance"].isin(appliances_for_hourly_plot)]

        # Create subplots for each appliance
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        # Create a subplot figure with one row per appliance
        fig_hourly = make_subplots(
            rows=len(appliances_for_hourly_plot),
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,  # Compact spacing between subplots
            subplot_titles=[f"Hourly Usage for {appliance.capitalize()}" for appliance in appliances_for_hourly_plot]
        )

        for idx, appliance in enumerate(appliances_for_hourly_plot):
            appliance_data = hourly_filtered_df[hourly_filtered_df["appliance"] == appliance]
            avg_hourly_usage = appliance_data.groupby("hour")["usage"].mean().reset_index()

            fig_hourly.add_trace(
                go.Scatter(
                    x=avg_hourly_usage["hour"],
                    y=avg_hourly_usage["usage"],
                    mode="lines",
                    name=appliance.capitalize()
                ),
                row=idx + 1,
                col=1
            )

        fig_hourly.update_layout(
            height=250 * len(appliances_for_hourly_plot),  # Adjust height dynamically
            title="Hourly Usage Trends by Appliance",
            xaxis=dict(title="Hour of Day", tickmode="linear", dtick=1),  # Increment by 1
            yaxis_title="Average Usage (kWh)",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False  # Disable legend since titles show appliance names
        )

        st.plotly_chart(fig_hourly, use_container_width=True, key="hourly_chart_subplots")

    with st.container(key="appliances-container"):
    # Heatmap: Appliances vs Usage
        st.markdown("### **_Appliances vs Usage Heatmap_**")

        # Adjust X-axis based on selection
        if x_axis_selector == "Hour":
            heatmap_data = filtered_df[filtered_df["appliance"].isin(appliances_to_display)].groupby(["appliance", "hour"])["usage"].mean().reset_index()
            x_axis_label = "hour"
        elif x_axis_selector == "Time of Day":
            heatmap_data = filtered_df[filtered_df["appliance"].isin(appliances_to_display)].groupby(["appliance", "time_of_day"])["usage"].mean().reset_index()
            x_axis_label = "time_of_day"
        else:  # Season
            heatmap_data = filtered_df[filtered_df["appliance"].isin(appliances_to_display)].groupby(["appliance", "season"])["usage"].mean().reset_index()
            x_axis_label = "season"

        # Create the heatmap
        fig_heatmap = px.density_heatmap(
            heatmap_data,
            x=x_axis_label,
            y="appliance",
            z="usage",
            color_continuous_scale="Viridis",
            title=f"Appliances vs {x_axis_selector} (Usage)",
            labels={x_axis_label: x_axis_selector, "appliance": "Appliance", "usage": "Avg Usage (kWh)"},
        )
        fig_heatmap.update_layout(
            xaxis=dict(title=x_axis_selector, tickmode="linear", dtick=1),  # Increment by 1
            yaxis=dict(title="Appliance"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=800  # Make the heatmap larger
        )

        # Larger Heatmap Display
        st.plotly_chart(fig_heatmap, use_container_width=True, key="heatmap_chart")


elif selected_tab == "Outliers Analysis":
    with st.container(key=""):
        st.sidebar.markdown("### Filters for Outliers Analysis")

        # Appliance Selector
        selected_appliances = st.sidebar.multiselect(
            "Select Appliances to Analyze",
            options=filtered_df["appliance"].unique(),
            default=filtered_df["appliance"].unique(),
            help="Filter the analysis by appliance."
        )

        # Filter data for the selected appliances
        outliers_filtered_df = filtered_df[filtered_df["appliance"].isin(selected_appliances)]

        # Chart 1: Outlier Percentage by Appliance
        st.markdown("### **_Outlier Percentage by Appliance_**")
        outlier_percentage = (
            outliers_filtered_df.groupby("appliance")["outlier"].mean() * 100
        ).reset_index()
        outlier_percentage.rename(columns={"outlier": "outlier_percentage"}, inplace=True)

        fig_outlier_percentage = px.bar(
            outlier_percentage,
            x="appliance",
            y="outlier_percentage",
            title="Percentage of Outliers by Appliance",
            labels={"appliance": "Appliance", "outlier_percentage": "Outlier Percentage (%)"}
        )
        fig_outlier_percentage.update_layout(
            xaxis=dict(title="Appliance"),
            yaxis=dict(title="Outlier Percentage (%)"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_outlier_percentage, use_container_width=True, key="outlier_percentage_chart")

    with st.container(key=""):
        # Chart 2: Periods Associated with Outliers
        st.markdown("### **_Periods Associated with Outliers_**")

        # Calculate percentage of outliers for each segment
        total_records = len(outliers_filtered_df)
        outliers_by_day = (
            outliers_filtered_df.groupby(outliers_filtered_df["date"].dt.day_name())["outlier"]
            .mean() * 100
        ).reset_index().rename(columns={"date": "day_name", "outlier": "outlier_percentage"})

        outliers_by_hour = (
            outliers_filtered_df.groupby("hour")["outlier"]
            .mean() * 100
        ).reset_index().rename(columns={"hour": "hour_of_day", "outlier": "outlier_percentage"})

        outliers_by_month = (
            outliers_filtered_df.groupby(outliers_filtered_df["date"].dt.month_name())["outlier"]
            .mean() * 100
        ).reset_index().rename(columns={"date": "month", "outlier": "outlier_percentage"})

        # Create plots for each period
        fig_day = px.bar(
            outliers_by_day,
            x="day_name",
            y="outlier_percentage",
            title="Outlier Percentage by Day of the Week",
            labels={"day_name": "Day of the Week", "outlier_percentage": "Outlier Percentage (%)"}
        )
        fig_hour = px.bar(
            outliers_by_hour,
            x="hour_of_day",
            y="outlier_percentage",
            title="Outlier Percentage by Hour of the Day",
            labels={"hour_of_day": "Hour of the Day", "outlier_percentage": "Outlier Percentage (%)"}
        )
        fig_month = px.bar(
            outliers_by_month,
            x="month",
            y="outlier_percentage",
            title="Outlier Percentage by Month",
            labels={"month": "Month", "outlier_percentage": "Outlier Percentage (%)"}
        )

        # Display charts in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.plotly_chart(fig_day, use_container_width=True, key="outlier_by_day_chart")
        with col2:
            st.plotly_chart(fig_hour, use_container_width=True, key="outlier_by_hour_chart")
        with col3:
            st.plotly_chart(fig_month, use_container_width=True, key="outlier_by_month_chart")



elif selected_tab == "Chat with an Expert":
    

    # Organized filters layout
    st.sidebar.subheader("Filter Data for Analysis")

    appliance_filter_llm = st.sidebar.multiselect(
        "Appliances",
        options=llm_df['appliance'].unique(),
        default=llm_df['appliance'].unique()
    )

    outlier_filter_llm = st.sidebar.selectbox(
        "Include Outliers?",
        options=["Yes", "No"],
        index=1
    )
    
    # Apply the filters
    filtered_llm_df = llm_df[
        (llm_df['city'].isin(city_filter)) &
        (llm_df['season'].isin(season_filter)) &
        (llm_df['appliance'].isin(appliance_filter_llm)) &
        ((llm_df['outlier'] == 0) if outlier_filter_llm == "No" else True)
    ]
    
    if filtered_llm_df.empty:
        st.warning("No data available for the selected filters. Please adjust your filters.")
    else:
        with st.container(key=""):
            st.subheader("Ask Anything")
            col1, col2 = st.columns([1, 2])

            with col1:
                st.write("#### Example questions:")
                st.write("- Compare energy usage between bedroom and living room")
                st.write("- Which appliance consumes the most energy?")
                st.write("- How does kitchenArea usage vary by season?")
                st.write("- What's the peak usage time for the bathroom?")
                st.write("- Can you generate a chart comparing appliances?")
            # Populate the second column
            with col2:
                data_context = create_data_context(filtered_llm_df)
                
                # Query interface
                question = st.text_input(
                    "Ask a question about the filtered energy usage (charts can be generated if needed):",
                    key="llm_question_input"  # Unique key for the text input
                )
                
                if st.button("Analyze"):
                    if question:
                        with st.spinner("Analyzing..."):
                            try:
                                answer = query_model(question, data_context)
                                st.write("### Analysis:")
                                st.write(answer)
                                
                                # Check if the LLM suggests a chart
                                if "chart" in question.lower():
                                    st.subheader("Generated Chart")
                                    # Example chart: Average usage by appliance
                                    avg_usage = filtered_llm_df.groupby('appliance')['usage'].mean().reset_index()
                                    fig = px.bar(
                                        avg_usage,
                                        x="appliance",
                                        y="usage",
                                        title="Average Usage by Appliance",
                                        labels={"appliance": "Appliance", "usage": "Usage (kWh)"}
                                    )
                                    st.plotly_chart(fig)
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
                    else:
                        st.warning("Please enter a question.")
                    if question:
                        with st.spinner("Analyzing..."):
                            try:
                                answer = query_model(question, data_context)
                                st.write("### Analysis:")
                                st.write(answer)
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
                    else:
                        st.warning("Please enter a question.")  