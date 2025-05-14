import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# Your data generation function (unchanged)
def generate_workshop_data(n_samples=9891):
    motor_type_weights = {
        'ALL-NEW-AEROX': 0.12, 'LEXI': 0.10, 'FREE-GO-S': 0.08, 'MIO': 0.15,
        'NMAX-NEO-S': 0.14, 'X-MAX-CONNEC': 0.08, 'ALL-NEW-NMAX': 0.10,
        'MIO-M3 125': 0.07, 'FAZZIO-NEO-HY': 0.05, 'FILANO-NEO': 0.04,
        'XSR 155': 0.06, 'MT-15': 0.05, 'R15': 0.04, 'VIXION': 0.02
    }
    motor_types = list(motor_type_weights.keys())
    
    service_type_weights = {
        'Rutin': 0.45, 'Repair': 0.25, 'Kupon Gratis I (DUA)': 0.10,
        'Kupon Gratis II (TIGA)': 0.06, 'Kupon Gratis III (TIGA)': 0.03, 'Jual Part': 0.01
    }
    service_types = list(service_type_weights.keys())

    start_date = datetime(2021, 1, 1)
    end_date = datetime(2024, 12, 31)
    dates = [start_date + timedelta(days=random.randint(0, (end_date - start_date).days)) 
             for _ in range(n_samples)]
    dates.sort()

    oil_change_intervals = {
        'ALL-NEW-AEROX': 3000, 'LEXI': 2800, 'FREE-GO-S': 2500, 'MIO': 2500,
        'NMAX-NEO-S': 3000, 'X-MAX-CONNEC': 3200, 'ALL-NEW-NMAX': 3000,
        'MIO-M3 125': 2500, 'FAZZIO-NEO-HY': 2800, 'FILANO-NEO': 2800,
        'XSR 155': 3500, 'MT-15': 3500, 'R15': 3500, 'VIXION': 3500
    }
    
    oil_change_months = 6

    data = {
        'Tgl FJB': [date.strftime('%d-%m-%Y') for date in dates],
        'No FJB': [f'FJB/{str(random.randint(1000, 9999))}/SBY/PL/{random.randint(1000, 9999)}' 
                   for _ in range(n_samples)],
        'Kode Cust': [f'SBY{random.randint(1000000, 9999999)}' for _ in range(n_samples)],
        'Kode Unit Motor': [random.choices(motor_types, weights=list(motor_type_weights.values()))[0] 
                           for _ in range(n_samples)],
        'Tahun Kendaraan': [],
        'Kilometer': [],
        'Jns Service': [],
        'Kondisi_Oli': [],
        'Kondisi_Rem': [],
        'Bulan_Terakhir_Ganti_Oli': [],
        'KM_Terakhir_Ganti_Oli': [],
        'Usia_Kendaraan': [],
        'Jumlah_Keluhan': []
    }

    for i in range(n_samples):
        current_date = datetime.strptime(data['Tgl FJB'][i], '%d-%m-%Y')
        max_year = min(current_date.year, 2024)
        tahun = random.randint(2018, max_year)
        usia = current_date.year - tahun
        motor_type = data['Kode Unit Motor'][i]
        
        usage_patterns = {
            'ALL-NEW-AEROX': random.uniform(1.5, 1.8), 'LEXI': random.uniform(1.3, 1.5),
            'FREE-GO-S': random.uniform(1.2, 1.4), 'MIO': random.uniform(1.3, 1.6),
            'NMAX-NEO-S': random.uniform(1.5, 1.9), 'X-MAX-CONNEC': random.uniform(1.6, 2.0),
            'ALL-NEW-NMAX': random.uniform(1.5, 1.9), 'MIO-M3 125': random.uniform(1.3, 1.6),
            'FAZZIO-NEO-HY': random.uniform(1.2, 1.5), 'FILANO-NEO': random.uniform(1.2, 1.5),
            'XSR 155': random.uniform(1.6, 2.0), 'MT-15': random.uniform(1.5, 2.0),
            'R15': random.uniform(1.6, 2.1), 'VIXION': random.uniform(1.6, 2.0)
        }
        
        yearly_km_base = random.randint(4500, 8500)
        multiplier = usage_patterns.get(motor_type, random.uniform(1.2, 1.6))
        km = max(usia * yearly_km_base * multiplier + random.randint(-400, 400), 0)
        
        recommended_km_interval = oil_change_intervals.get(motor_type, 3000)
        
        oil_maintenance_habit = random.choices(
            ['excellent', 'good', 'average', 'poor', 'neglected'],
            weights=[0.15, 0.30, 0.35, 0.15, 0.05]
        )[0]
        
        if oil_maintenance_habit == 'excellent':
            km_since_oil = int(random.uniform(0.5, 0.8) * recommended_km_interval)
            months_since_oil = int(random.uniform(0.5, 0.8) * oil_change_months)
        elif oil_maintenance_habit == 'good':
            km_since_oil = int(random.uniform(0.7, 0.9) * recommended_km_interval)
            months_since_oil = int(random.uniform(0.7, 0.9) * oil_change_months)
        elif oil_maintenance_habit == 'average':
            km_since_oil = int(random.uniform(0.8, 1.1) * recommended_km_interval)
            months_since_oil = int(random.uniform(0.8, 1.1) * oil_change_months)
        elif oil_maintenance_habit == 'poor':
            km_since_oil = int(random.uniform(1.1, 1.5) * recommended_km_interval)
            months_since_oil = int(random.uniform(1.1, 1.5) * oil_change_months)
        else:
            km_since_oil = int(random.uniform(1.5, 2.2) * recommended_km_interval)
            months_since_oil = int(random.uniform(1.5, 2.0) * oil_change_months)
        
        km_percentage = km_since_oil / recommended_km_interval
        month_percentage = months_since_oil / oil_change_months
        oil_factor = max(km_percentage, month_percentage)
        
        if oil_factor < 0.6:
            oli_condition = 'Baik'
        elif oil_factor < 0.9:
            oli_condition = 'Cukup'
        elif oil_factor < 1.3:
            oli_condition = 'Perlu Ganti'
        else:
            oli_condition = 'Kritis'
        
        if usia <= 1 and km < 10000:
            rem_condition = 'Baik'
        elif (usia <= 2 and km < 18000) or (usia <= 1 and km < 15000):
            rem_condition = 'Cukup'
        elif (usia <= 3 and km < 25000) or (usia <= 2 and km < 20000):
            rem_condition = 'Perlu Penyetelan'
        else:
            rem_condition = 'Perlu Penggantian'
        
        if oli_condition == 'Kritis':
            oil_complaints = random.choices([2, 3, 4], weights=[0.15, 0.45, 0.40])[0]
        elif oli_condition == 'Perlu Ganti':
            oil_complaints = random.choices([1, 2, 3], weights=[0.30, 0.50, 0.20])[0]
        elif oli_condition == 'Cukup':
            oil_complaints = random.choices([0, 1], weights=[0.75, 0.25])[0]
        else:
            oil_complaints = 0
        
        if rem_condition == 'Perlu Penggantian':
            brake_complaints = random.choices([1, 2, 3], weights=[0.20, 0.50, 0.30])[0]
        elif rem_condition == 'Perlu Penyetelan':
            brake_complaints = random.choices([0, 1, 2], weights=[0.30, 0.50, 0.20])[0]
        elif rem_condition == 'Cukup':
            brake_complaints = random.choices([0, 1], weights=[0.80, 0.20])[0]
        else:
            brake_complaints = 0
        
        other_complaints = random.choices([0, 1, 2], weights=[0.7, 0.25, 0.05])[0]
        total_complaints = min(oil_complaints + brake_complaints + other_complaints, 5)
        
        if oli_condition == 'Kritis':
            repair_chance = 0.95
        elif oli_condition == 'Perlu Ganti':
            repair_chance = 0.85
        elif rem_condition == 'Perlu Penggantian':
            repair_chance = 0.90
        elif rem_condition == 'Perlu Penyetelan':
            repair_chance = 0.75
        elif total_complaints >= 3:
            repair_chance = 0.90
        elif total_complaints >= 2:
            repair_chance = 0.80
        elif total_complaints == 1:
            repair_chance = 0.40
        else:
            repair_chance = 0.10
            
        if km > 20000 or usia >= 3:
            repair_chance += 0.15
            repair_chance = min(repair_chance, 0.98)
            
        if random.random() < repair_chance:
            service = 'Repair'
        else:
            if km < 1000 and usia <= 1:
                service = random.choices(
                    ['Rutin', 'Kupon Gratis I (DUA)', 'Kupon Gratis II (TIGA)', 'Kupon Gratis III (TIGA)'],
                    weights=[0.3, 0.5, 0.15, 0.05]
                )[0]
            else:
                service = random.choices(
                    ['Rutin', 'Kupon Gratis I (DUA)', 'Kupon Gratis II (TIGA)', 'Jual Part'],
                    weights=[0.85, 0.08, 0.05, 0.02]
                )[0]
        
        data['Tahun Kendaraan'].append(tahun)
        data['Kilometer'].append(int(round(km, 0)))
        data['Jns Service'].append(service)
        data['Kondisi_Oli'].append(oli_condition)
        data['Kondisi_Rem'].append(rem_condition)
        data['Bulan_Terakhir_Ganti_Oli'].append(months_since_oil)
        data['KM_Terakhir_Ganti_Oli'].append(km_since_oil)
        data['Usia_Kendaraan'].append(usia)
        data['Jumlah_Keluhan'].append(total_complaints)

    df = pd.DataFrame(data)
    
    df['KM_Per_Tahun'] = (df['Kilometer'] / df['Usia_Kendaraan'].replace(0, 1)).astype(int)
    
    df['Total_Usage_Score'] = (
        (df['Kilometer'] / 5000).astype(int) * 1.0 + 
        df['Usia_Kendaraan'] * 1.5 +
        (df['KM_Terakhir_Ganti_Oli'] / 800).astype(int) * 2.0 +
        df['Bulan_Terakhir_Ganti_Oli'] * 0.8 +
        df['Jumlah_Keluhan'] * 2.0
    )
    
    oil_condition_map = {'Baik': 0, 'Cukup': 1, 'Perlu Ganti': 2, 'Kritis': 3}
    brake_condition_map = {'Baik': 0, 'Cukup': 1, 'Perlu Penyetelan': 2, 'Perlu Penggantian': 3}
    
    df['Kondisi_Oli_Score'] = df['Kondisi_Oli'].map(oil_condition_map)
    df['Kondisi_Rem_Score'] = df['Kondisi_Rem'].map(brake_condition_map)
    
    df['Oil_Efficiency'] = df['KM_Terakhir_Ganti_Oli'] / (df['Bulan_Terakhir_Ganti_Oli'].replace(0, 1))
    recommended_intervals = df['Kode Unit Motor'].map(oil_change_intervals)
    df['Oil_Change_Urgency'] = df['KM_Terakhir_Ganti_Oli'] / recommended_intervals
    
    df['Oil_Based_Repair_Risk'] = (
        (df['Kondisi_Oli'] == 'Kritis') * 0.95 + 
        (df['Kondisi_Oli'] == 'Perlu Ganti') * 0.75 + 
        (df['Kondisi_Oli'] == 'Cukup') * 0.3 + 
        (df['Kondisi_Oli'] == 'Baik') * 0.1
    )
    
    df['Maintenance_Urgency'] = (
        df['Oil_Change_Urgency'] * 2.5 +
        df['Kondisi_Oli_Score'] * 1.8 +
        df['Kondisi_Rem_Score'] * 1.2 +
        df['Jumlah_Keluhan'] * 0.8
    )
    
    df['Oil_Rem_Interaction'] = df['Kondisi_Oli_Score'] * df['Kondisi_Rem_Score']
    df['Usage_Keluhan_Interaction'] = df['Total_Usage_Score'] * df['Jumlah_Keluhan']
    
    return df

# Generate data
df = generate_workshop_data(9891)

# ====== ENHANCED FEATURE ENGINEERING ======

# 1. Create new categorical features
df['KM_Category'] = pd.cut(df['Kilometer'], 
                          bins=[0, 5000, 10000, 20000, 30000, 100000], 
                          labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

df['Motor_Performance_Class'] = 'Standard'
sport_models = ['XSR 155', 'MT-15', 'R15', 'VIXION', 'ALL-NEW-AEROX']
luxury_models = ['X-MAX-CONNEC', 'ALL-NEW-NMAX', 'NMAX-NEO-S']
economy_models = ['MIO', 'MIO-M3 125', 'FREE-GO-S', 'FAZZIO-NEO-HY', 'FILANO-NEO', 'LEXI']

df.loc[df['Kode Unit Motor'].isin(sport_models), 'Motor_Performance_Class'] = 'Sport'
df.loc[df['Kode Unit Motor'].isin(luxury_models), 'Motor_Performance_Class'] = 'Luxury'
df.loc[df['Kode Unit Motor'].isin(economy_models), 'Motor_Performance_Class'] = 'Economy'

# 2. Enhanced scores and features
# Oil change patterns relative to recommended interval
oil_change_intervals = {
    'ALL-NEW-AEROX': 3000, 'LEXI': 2800, 'FREE-GO-S': 2500, 'MIO': 2500,
    'NMAX-NEO-S': 3000, 'X-MAX-CONNEC': 3200, 'ALL-NEW-NMAX': 3000,
    'MIO-M3 125': 2500, 'FAZZIO-NEO-HY': 2800, 'FILANO-NEO': 2800,
    'XSR 155': 3500, 'MT-15': 3500, 'R15': 3500, 'VIXION': 3500
}
recommended_intervals = df['Kode Unit Motor'].map(oil_change_intervals)

df['Maintenance_Delay_Score'] = (df['KM_Terakhir_Ganti_Oli'] - recommended_intervals) / 1000
df['Maintenance_Delay_Score'] = df['Maintenance_Delay_Score'].clip(lower=-3, upper=5)

# Interaction between age and usage
df['Age_Usage_Impact'] = np.sqrt(df['Usia_Kendaraan']) * np.log1p(df['Kilometer'] / 1000)

# 3. More sophisticated interaction terms
df['Oil_Critical_High_KM'] = ((df['Kondisi_Oli'] == 'Kritis') & (df['Kilometer'] > 15000)).astype(int)
df['Rem_Critical_High_KM'] = ((df['Kondisi_Rem'] == 'Perlu Penggantian') & (df['Kilometer'] > 15000)).astype(int)
df['Dual_Critical_Components'] = ((df['Kondisi_Oli_Score'] >= 2) & (df['Kondisi_Rem_Score'] >= 2)).astype(int)

# Usage intensity metrics
df['Usage_Intensity'] = df['KM_Per_Tahun'] / 5000  # Normalized per 5000km standard
df['Maintenance_Quality'] = 5 - (df['Oil_Change_Urgency'] * 2 + df['Kondisi_Rem_Score'] / 2)
df['Maintenance_Quality'] = df['Maintenance_Quality'].clip(lower=0, upper=5)

# Sophisticated component interactions
df['Overall_Vehicle_Health'] = 10 - (
    df['Kondisi_Oli_Score'] * 1.5 + 
    df['Kondisi_Rem_Score'] * 1.2 + 
    np.sqrt(df['Jumlah_Keluhan']) * 2.0 +
    df['Maintenance_Delay_Score'] * 0.8
)
df['Overall_Vehicle_Health'] = df['Overall_Vehicle_Health'].clip(lower=0, upper=10)

# Risk scores
df['Repair_Risk_Score'] = (
    df['Jumlah_Keluhan'] * 1.8 + 
    df['Kondisi_Oli_Score'] * 1.5 + 
    df['Kondisi_Rem_Score'] * 1.2 + 
    (df['Kilometer'] > 20000).astype(int) * 1.0 +
    (df['Usia_Kendaraan'] >= 3).astype(int) * 1.2 +
    df['Oil_Critical_High_KM'] * 2.5 +
    df['Rem_Critical_High_KM'] * 2.0 +
    df['Dual_Critical_Components'] * 3.0
)

# One-hot encode categorical variables
motor_class_dummies = pd.get_dummies(df['Motor_Performance_Class'], prefix='Class', drop_first=False)
km_category_dummies = pd.get_dummies(df['KM_Category'], prefix='KM_Cat', drop_first=False)
df = pd.concat([df, motor_class_dummies, km_category_dummies], axis=1)

# Check which columns were actually created
motor_class_columns = [col for col in df.columns if col.startswith('Class_')]
km_category_columns = [col for col in df.columns if col.startswith('KM_Cat_')]

# 4. Create type-specific maintenance patterns
df['Sport_High_KM'] = ((df['Motor_Performance_Class'] == 'Sport') & (df['Kilometer'] > 15000)).astype(int)
df['Economy_Poor_Maintenance'] = ((df['Motor_Performance_Class'] == 'Economy') & 
                                (df['Maintenance_Quality'] < 2)).astype(int)
df['Luxury_Oil_Issues'] = ((df['Motor_Performance_Class'] == 'Luxury') & 
                         (df['Kondisi_Oli_Score'] >= 2)).astype(int)

# Create target variable
y = (df['Jns Service'] == 'Repair').astype(int)

# Select features for model - now dynamically getting column names
features = [
    # Basic metrics
    'Kilometer', 'KM_Terakhir_Ganti_Oli', 'Bulan_Terakhir_Ganti_Oli', 
    'Usia_Kendaraan', 'Jumlah_Keluhan', 'KM_Per_Tahun',
    
    # Component conditions
    'Kondisi_Oli_Score', 'Kondisi_Rem_Score',
    
    # Advanced metrics
    'Oil_Efficiency', 'Oil_Change_Urgency', 'Oil_Based_Repair_Risk',
    'Maintenance_Urgency', 'Oil_Rem_Interaction', 'Usage_Keluhan_Interaction',
    'Total_Usage_Score',
    
    # New features
    'Maintenance_Delay_Score', 'Age_Usage_Impact', 'Oil_Critical_High_KM',
    'Rem_Critical_High_KM', 'Dual_Critical_Components', 'Usage_Intensity',
    'Maintenance_Quality', 'Overall_Vehicle_Health', 'Repair_Risk_Score',
    'Sport_High_KM', 'Economy_Poor_Maintenance', 'Luxury_Oil_Issues',
]

# Add the actual one-hot encoded columns from dataframe
features.extend(motor_class_columns)
features.extend(km_category_columns)

X = df[features]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create a pipeline for preprocessing and modeling
pipeline = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
    ('model', XGBClassifier(random_state=42))
])

# Set up parameters for grid search
param_grid = {
    'model__n_estimators': [200, 300],
    'model__learning_rate': [0.05, 0.1],
    'model__max_depth': [6, 8],
    'model__min_child_weight': [1, 3],
    'model__gamma': [0, 0.1],
    'model__subsample': [0.8, 1.0],
    'model__colsample_bytree': [0.8, 1.0]
}

# Perform grid search
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Evaluate on test set
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Cross-validation score
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')

# Get feature importances (from the model after polynomial features)
# Extract feature names after polynomial transformation
poly_features = best_model.named_steps['poly']
feature_names = poly_features.get_feature_names_out(features)

# Get the model
xgb_model = best_model.named_steps['model']
importances = xgb_model.feature_importances_

# Create a DataFrame for feature importances
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

# Print results
print(f"\nModel Accuracy: {accuracy:.4f}")
print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"Best Parameters: {grid_search.best_params_}")

print("\nTop 20 Feature Importances:")
for feature, importance in feature_importance.head(20).values:
    print(f"{feature}: {importance:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Regular', 'Repair']))

# Try ensemble modeling for potentially even better results
rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
gb_model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42)
nn_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42)

# Create an ensemble of models
models = [
    ('XGBoost', xgb_model),
    ('RandomForest', rf_model),
    ('GradientBoosting', gb_model),
    ('NeuralNetwork', nn_model)
]

X_train_processed = best_model.named_steps['poly'].transform(best_model.named_steps['scaler'].transform(X_train))
X_test_processed = best_model.named_steps['poly'].transform(best_model.named_steps['scaler'].transform(X_test))
X_train_resampled, y_train_resampled = best_model.named_steps['smote'].fit_resample(X_train_processed, y_train)

print("\nEnsemble Model Performance:")
for name, model in models:
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test_processed)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.4f}")

# Save the enhanced data
df.to_csv('enhanced_indoperkasa.csv', index=False)

# Final insights
print("\nDistribution of Complaints by Service Type:")
print(df.groupby('Jns Service')['Jumlah_Keluhan'].describe())

print("\nDistribution of Oil Condition by Service Type:")
print(pd.crosstab(df['Jns Service'], df['Kondisi_Oli'], normalize='index').round(3))

print("\nRelationship between Overall Vehicle Health and Repair Probability:")
health_bins = [0, 2, 4, 6, 8, 10]
df['Health_Group'] = pd.cut(df['Overall_Vehicle_Health'], bins=health_bins)
health_analysis = df.groupby('Health_Group')['Jns Service'].apply(lambda x: (x == 'Repair').mean())
print(health_analysis)

print("\nRelationship between Oil Change Urgency and Repair Probability:")
urgency_bins = [0, 0.6, 0.9, 1.2, 1.5, 5.0]
df['Urgency_Group'] = pd.cut(df['Oil_Change_Urgency'], bins=urgency_bins)
urgency_analysis = df.groupby('Urgency_Group')['Jns Service'].apply(lambda x: (x == 'Repair').mean())
print(urgency_analysis)