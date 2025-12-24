# Prophet.py - Complete Method-by-Method Explanation for 1-Hour Presentation
## AI Analytics for Observability - DevOps Team Knowledge Transfer

---

## 📋 **PRESENTATION STRUCTURE (60 Minutes)**

**0-5 min**: Introduction & Overview  
**5-15 min**: Data Preparation & Aggregation Methods  
**15-25 min**: Prophet Forecasting & Anomaly Detection  
**25-35 min**: Feature Engineering & LSTM Deep Learning  
**35-45 min**: Unsupervised Learning Methods (Isolation Forest, SVM, Clustering)  
**45-55 min**: Ensemble Detection & AI Report Generation  
**55-60 min**: Q&A & Next Steps

---

# 🎯 **SECTION 1: INTRODUCTION (0-5 min)**

## **What This Script Does**

This `prophet.py` script is a **complete AI-powered anomaly detection pipeline** for DevOps observability. It takes raw cybersecurity logs and automatically:

1. ✅ Aggregates data into hourly metrics
2. ✅ Detects anomalies using 5 different ML algorithms
3. ✅ Identifies which features caused each anomaly
4. ✅ Finds recurring patterns (daily/hourly trends)
5. ✅ Generates professional PDF reports with AI explanations

## **Why Multiple Methods?**

Each detection method has strengths and weaknesses:

| Method | Best For | Limitation |
|--------|----------|------------|
| Prophet | Seasonal patterns, forecasting | Misses sudden attacks |
| LSTM | Complex temporal patterns | Slow training |
| Isolation Forest | Unknown anomalies | May flag rare but normal events |
| One-Class SVM | High-dimensional data | Computationally expensive |
| K-Means Clustering | Pattern grouping | Needs predefined cluster count |

**Ensemble approach = Higher confidence when multiple methods agree**

---

# 📊 **SECTION 2: DATA PREPARATION & AGGREGATION (5-15 min)**

## **METHOD 1: Load Raw Logs (Lines 6-9)**

```python
df = pd.read_csv(r'/home/rle2004/model/advanced_cybersecurity_data.csv')
```

### **What It Does**
- Loads raw cybersecurity logs from CSV file
- Expected columns: `Timestamp`, `IP_Address`, `Session_ID`, `Request_Type`, `Status_Code`

### **Why Important**
- Raw logs are too granular (thousands of entries per minute)
- Need aggregation for pattern detection
- Individual requests don't show trends

---

## **METHOD 2: Convert Timestamps (Lines 12-13)**

```python
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.set_index('Timestamp', inplace=True)
```

### **What It Does**
- Converts string timestamps to datetime objects
- Sets timestamp as the index for time-series operations

### **Why Important**
- Enables time-based operations (resampling, rolling windows)
- Required for Prophet and LSTM models
- Allows filtering by date ranges

---

## **METHOD 3: Hourly Aggregation (Lines 18-23)**

```python
resampled = df.resample('1H').agg({
    'IP_Address': pd.Series.nunique,           # Count unique IPs
    'Session_ID': pd.Series.nunique,           # Count unique sessions
    'Request_Type': 'count',                   # Total requests
    'Status_Code': lambda x: (x >= 400).sum()  # Count errors
})
```

### **What It Does**
- Groups all data into 1-hour buckets
- Calculates 4 key metrics per hour:
  1. **Unique IPs**: How many different IP addresses made requests
  2. **Unique Sessions**: How many active user sessions
  3. **Total Requests**: Total number of HTTP requests
  4. **Error Count**: Number of failed requests (4xx, 5xx status codes)

### **Why Important**
- **Reduces noise**: From thousands of logs to ~24 data points per day
- **Reveals patterns**: Hourly trends are easier to analyze than per-second data
- **Performance**: ML models train faster on aggregated data
- **Anomaly detection**: Easier to spot "unusual hours" than "unusual seconds"

### **Example**
```
Before aggregation (raw logs):
2025-01-15 14:23:01, 192.168.1.1, sess_001, GET, 200
2025-01-15 14:23:02, 192.168.1.2, sess_002, POST, 201
2025-01-15 14:23:03, 192.168.1.1, sess_001, GET, 404
... (1000 more rows)

After aggregation (hourly):
2025-01-15 14:00:00, Unique_IPs=150, Unique_Sessions=200, Total_Requests=1200, Error_Count=45
```

---

## **METHOD 4: Rename Columns (Lines 26-31)**

```python
resampled.rename(columns={
    'IP_Address': 'Unique_IPs',
    'Session_ID': 'Unique_Sessions',
    'Request_Type': 'Total_Requests',
    'Status_Code': 'Error_Count'
}, inplace=True)
```

### **What It Does**
- Renames columns to be more descriptive

### **Why Important**
- **Clarity**: "Unique_IPs" is clearer than "IP_Address" after aggregation
- **Prevents confusion**: Original column names don't reflect aggregated meaning
- **Better reports**: Clear column names in PDF reports

---

## **METHOD 5: Fill Missing Data (Line 34)**

```python
resampled = resampled.fillna(0)
```

### **What It Does**
- Replaces NaN (missing) values with 0

### **Why Important**
- **Prevents errors**: ML models can't handle NaN values
- **Logical default**: If no data for an hour, assume 0 requests
- **Continuity**: Maintains continuous time series

---

## **METHOD 6: Save Aggregated Data (Lines 37-40)**

```python
resampled = resampled.reset_index()
resampled.to_csv('aggregated_logs.csv', index=False)
print("Aggregated data saved to 'aggregated_logs.csv'")
```

### **What It Does**
- Resets index to make Timestamp a regular column
- Saves to CSV file

### **Why Important**
- **Checkpoint**: Can restart analysis from here without re-aggregating
- **Debugging**: Can inspect aggregated data manually
- **Sharing**: Other teams can use this cleaned data

---

## **METHOD 7: Visualization (Lines 54-71)**

```python
plt.figure(figsize=(12, 6))
plt.plot(df['Timestamp'], df['Total_Requests'], label='Total Requests')
plt.plot(df['Timestamp'], df['Error_Count'], label='Error Count', color='red')
plt.xlabel('Time')
plt.ylabel('Count')
plt.title('Requests and Errors Over Time')
plt.legend()
plt.show()
```

### **What It Does**
- Creates a line graph showing requests and errors over time

### **Why Important**
- **Visual inspection**: Quickly spot obvious spikes or drops
- **Pattern recognition**: See daily/weekly cycles
- **Baseline understanding**: Know what "normal" looks like before ML analysis

### **What to Look For**
1. **Daily patterns**: Traffic peaks during business hours
2. **Weekly patterns**: Lower traffic on weekends
3. **Spikes**: Sudden increases (DDoS attacks, viral content)
4. **Drops**: Sudden decreases (outages, bugs)

---

# 🔮 **SECTION 3: PROPHET FORECASTING (15-25 min)**

## **METHOD 8: Prepare Data for Prophet (Lines 91-101)**

```python
from prophet import Prophet

prophet_df = df[['Timestamp', 'Total_Requests']].rename(columns={
    'Timestamp': 'ds',           # Prophet requires 'ds' for dates
    'Total_Requests': 'y'        # Prophet requires 'y' for values
})
```

### **What It Does**
- Converts data to Prophet's required format
- `ds` = date/timestamp column
- `y` = value to forecast

### **Why Important**
- **Prophet requirement**: Won't work with other column names
- **Single metric**: Prophet forecasts one metric at a time
- **Simplicity**: Focus on most important metric (Total_Requests)

---

## **METHOD 9: Initialize Prophet Model (Lines 105-106)**

```python
model = Prophet(daily_seasonality=True, weekly_seasonality=True)
```

### **What It Does**
- Creates a Prophet model with:
  - **Daily seasonality**: Learns patterns within each day (e.g., peak at 2 PM)
  - **Weekly seasonality**: Learns patterns across the week (e.g., lower on weekends)

### **Why Important**
- **Captures patterns**: Most systems have predictable daily/weekly cycles
- **Accurate forecasts**: Model learns "normal" is 1000 requests at 2 PM on Tuesday
- **Anomaly detection**: Deviations from learned patterns = anomalies

### **How Seasonality Works**
```
Daily pattern learned:
00:00 - 06:00: Low traffic (100 requests/hour)
06:00 - 09:00: Morning spike (500 requests/hour)
09:00 - 17:00: Business hours (1000 requests/hour)
17:00 - 00:00: Evening decline (300 requests/hour)

Weekly pattern learned:
Mon-Fri: High traffic (1000 avg)
Sat-Sun: Low traffic (300 avg)
```

---

## **METHOD 10: Train Prophet Model (Line 108)**

```python
model.fit(prophet_df)
```

### **What It Does**
- Trains the model on historical data
- Learns trend, seasonality, and holidays

### **Why Important**
- **Pattern learning**: Model now "knows" what normal looks like
- **Forecasting ability**: Can predict future values
- **Anomaly baseline**: Deviations from predictions = anomalies

### **What Prophet Learns**
1. **Trend**: Overall increase/decrease over time
2. **Daily seasonality**: Hourly patterns within a day
3. **Weekly seasonality**: Day-of-week patterns
4. **Changepoints**: Where trends shift

---

## **METHOD 11: Generate Forecast (Lines 112-116)**

```python
future = model.make_future_dataframe(periods=48, freq='h')
forecast = model.predict(future)
```

### **What It Does**
- Creates a dataframe for next 48 hours
- Generates predictions with confidence intervals

### **Why Important**
- **Future visibility**: Know what to expect in next 2 days
- **Capacity planning**: Prepare for predicted traffic spikes
- **Proactive alerts**: Warn before problems occur

### **Forecast Output**
```
ds                  yhat    yhat_lower  yhat_upper
2025-01-15 14:00   1000    800         1200
2025-01-15 15:00   1100    900         1300
2025-01-15 16:00   1050    850         1250
```

- `yhat`: Predicted value
- `yhat_lower`: Lower bound (95% confidence)
- `yhat_upper`: Upper bound (95% confidence)

---

## **METHOD 12: Detect Anomalies (Lines 122-132)**

```python
merged = pd.merge(prophet_df, forecast, on='ds', how='left')
merged['anomaly'] = (merged['y'] < merged['yhat_lower']) | (merged['y'] > merged['yhat_upper'])
```

### **What It Does**
- Compares actual values to forecast confidence interval
- Flags anomalies when actual is outside the interval

### **Why Important**
- **Automatic detection**: No manual threshold setting
- **Context-aware**: Considers time of day and day of week
- **Confidence-based**: Only flags significant deviations

### **Anomaly Logic**
```
If actual < yhat_lower:
    → Anomaly: "Below expected range" (possible outage, bug)
    
If actual > yhat_upper:
    → Anomaly: "Above expected range" (possible attack, viral content)
    
If yhat_lower ≤ actual ≤ yhat_upper:
    → Normal
```

### **Example**
```
Time: 2025-01-15 14:00 (Tuesday afternoon)
Predicted: 1000 requests (range: 800-1200)
Actual: 5000 requests
→ ANOMALY: 5000 > 1200 (possible DDoS attack)

Time: 2025-01-15 03:00 (Tuesday night)
Predicted: 100 requests (range: 50-150)
Actual: 120 requests
→ NORMAL: 50 ≤ 120 ≤ 150
```

---

## **METHOD 13: Explain Anomalies (Lines 163-178)**

```python
def explain_anomaly(row):
    if row['y'] < row['yhat_lower']:
        return 'Below expected range'
    elif row['y'] > row['yhat_upper']:
        return 'Above expected range'
    else:
        return 'Normal'

merged['anomaly_reason'] = merged.apply(lambda row: explain_anomaly(row), axis=1)
```

### **What It Does**
- Adds human-readable explanation for each anomaly

### **Why Important**
- **Actionable insights**: "Above expected" suggests different action than "Below expected"
- **Report clarity**: Management understands "Above expected range" better than "y > yhat_upper"
- **Debugging**: Quickly understand why something was flagged

---

## **METHOD 14: Visualize Prophet Results (Lines 181-209)**

```python
plt.figure(figsize=(12, 6))
plt.plot(merged['ds'], merged['y'], label='Actual', color='black')
plt.plot(merged['ds'], merged['yhat'], label='Forecast', color='blue')
plt.fill_between(merged['ds'], merged['yhat_lower'], merged['yhat_upper'], color='blue', alpha=0.2)
plt.scatter(anomalies['ds'], anomalies['y'], color='red', label='Anomalies', marker='o')
```

### **What It Does**
- Plots actual vs predicted values
- Shades confidence interval
- Highlights anomalies in red

### **Why Important**
- **Visual validation**: See if anomalies make sense
- **Pattern confirmation**: Verify model learned correct seasonality
- **Communication**: Easy to show stakeholders

---

# 🧠 **SECTION 4: FEATURE ENGINEERING (25-30 min)**

## **METHOD 15: Calculate Error Rate (Line 225)**

```python
df['Error_Rate'] = df['Error_Count'] / df['Total_Requests'].replace(0, 1)
```

### **What It Does**
- Calculates percentage of failed requests
- Prevents division by zero with `.replace(0, 1)`

### **Why Important**
- **Normalized metric**: 10 errors out of 100 requests (10%) is worse than 10 errors out of 10,000 (0.1%)
- **Service health indicator**: Rising error rate = degrading service
- **Anomaly detection**: Sudden error rate spike = critical issue

### **Example**
```
Hour 1: 50 errors / 1000 requests = 5% error rate (normal)
Hour 2: 50 errors / 100 requests = 50% error rate (ANOMALY - service degradation)
```

---

## **METHOD 16: Calculate Request Rate (Line 226)**

```python
df['Request_Rate'] = df['Total_Requests'] / 60
```

### **What It Does**
- Converts hourly requests to requests per minute

### **Why Important**
- **Standardized metric**: Easier to compare across different time windows
- **Capacity planning**: Know if infrastructure can handle X requests/minute
- **DDoS detection**: Sudden spike in request rate = potential attack

---

## **METHOD 17: Calculate Session Length (Line 227)**

```python
df['Session_Per_Request'] = df['Unique_Sessions'] / df['Total_Requests'].replace(0, 1)
```

### **What It Does**
- Calculates average requests per session

### **Why Important**
- **Bot detection**: Bots typically have very short sessions (1-2 requests)
- **User behavior**: Normal users have longer sessions (10-20 requests)
- **Anomaly indicator**: Sudden change suggests automated activity

### **Example**
```
Normal hour: 200 sessions / 2000 requests = 0.1 (10 requests per session)
Bot attack: 1000 sessions / 1100 requests = 0.91 (1.1 requests per session) → ANOMALY
```

---

## **METHOD 18: One-Hot Encode Request Types (Lines 274-275)**

```python
top_request_encoded = pd.get_dummies(df['Top_Request_Type'], prefix='Request')
df = pd.concat([df, top_request_encoded], axis=1)
```

### **What It Does**
- Converts categorical data (GET, POST, PUT, DELETE) to binary columns
- Creates columns: `Request_GET`, `Request_POST`, `Request_PUT`, `Request_DELETE`

### **Why Important**
- **ML requirement**: Most algorithms need numerical input
- **Pattern detection**: Unusual mix of request types = anomaly
- **Attack signatures**: Some attacks use specific request types

### **Example**
```
Before encoding:
Top_Request_Type = "GET"

After encoding:
Request_GET = 1
Request_POST = 0
Request_PUT = 0
Request_DELETE = 0
```

---

## **METHOD 19: Z-Score Anomaly Scoring (Lines 279-281)**

```python
df['Total_Requests_z'] = zscore(df['Total_Requests'])
df['Error_Rate_z'] = zscore(df['Error_Rate'])
df['Anomaly_Score'] = np.sqrt(df['Total_Requests_z']**2 + df['Error_Rate_z']**2)
```

### **What It Does**
- Calculates z-scores (standard deviations from mean)
- Combines multiple z-scores into single anomaly score

### **Why Important**
- **Standardized metric**: Compare different features on same scale
- **Combined risk**: High requests + high errors = higher anomaly score
- **Threshold setting**: Anomaly_Score > 3 = likely anomaly

### **How Z-Score Works**
```
Mean Total_Requests = 1000
Std Dev = 200

Hour 1: 1200 requests → z-score = (1200-1000)/200 = 1.0 (normal)
Hour 2: 2000 requests → z-score = (2000-1000)/200 = 5.0 (anomaly)
```

---

## **METHOD 20: Save Enhanced Dataset (Lines 284-287)**

```python
df.to_csv('enhanced_logs.csv', index=False)
print("Feature engineering completed. Enhanced dataset saved as 'enhanced_logs.csv'.")
```

### **What It Does**
- Saves dataset with all engineered features

### **Why Important**
- **Checkpoint**: Can skip feature engineering on subsequent runs
- **Model input**: This is what ML models will use
- **Analysis**: Can explore features in Excel/Jupyter

---

# 🤖 **SECTION 5: LSTM DEEP LEARNING (30-35 min)**

## **METHOD 21: Create Sequences (Lines 325-332)**

```python
sequence_length = 10

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]  # Past 10 hours
        y = data[i+seq_length]    # Next hour
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)
```

### **What It Does**
- Converts time series into sequences
- Uses past 10 hours to predict next hour

### **Why Important**
- **Temporal context**: LSTM needs to see history to learn patterns
- **Pattern learning**: "If last 10 hours look like X, next hour will be Y"
- **Anomaly detection**: Unusual sequences = anomalies

### **Example**
```
Input sequence (past 10 hours):
[1000, 1050, 1100, 1200, 1150, 1100, 1050, 1000, 950, 900]

Expected output (next hour):
850

Actual output:
2000 → ANOMALY (LSTM expected 850, got 2000)
```

---

## **METHOD 22: Build LSTM Model (Lines 346-350)**

```python
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(sequence_length, len(features))))
model.add(Dense(len(features)))
model.compile(optimizer='adam', loss='mse')
```

### **What It Does**
- Creates a neural network with:
  - **LSTM layer**: 64 neurons, learns temporal patterns
  - **Dense layer**: Outputs predictions for all features
  - **MSE loss**: Measures prediction error

### **Why Important**
- **Complex patterns**: Can learn non-linear relationships
- **Multi-feature**: Predicts all features simultaneously
- **Adaptive**: Continuously improves with more data

### **How LSTM Works**
```
LSTM has "memory" that remembers important past events:

Hour 1-10: Normal traffic pattern
Hour 11: Sudden spike
→ LSTM remembers: "After normal pattern, spike is unusual"

Hour 1-10: Gradual increase
Hour 11: Continued increase
→ LSTM remembers: "This is a trend, not an anomaly"
```

---

## **METHOD 23: Train LSTM (Line 353)**

```python
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)
```

### **What It Does**
- Trains model for 50 epochs (50 passes through data)
- Uses 10% of data for validation
- Batch size of 32 (processes 32 sequences at a time)

### **Why Important**
- **Learning**: Model adjusts weights to minimize prediction error
- **Validation**: Prevents overfitting (memorizing training data)
- **Convergence**: 50 epochs usually enough for stable predictions

---

## **METHOD 24: Detect Anomalies with LSTM (Lines 405-414)**

```python
y_pred = model.predict(X)
mse = np.mean(np.square(y - y_pred), axis=1)
threshold = np.mean(mse) + 3 * np.std(mse)
anomalies = mse > threshold
```

### **What It Does**
- Predicts values for all sequences
- Calculates Mean Squared Error (prediction error)
- Sets threshold at mean + 3 standard deviations
- Flags sequences with error above threshold

### **Why Important**
- **Adaptive threshold**: Automatically adjusts to data distribution
- **Confidence-based**: 3 std devs = 99.7% confidence
- **Multi-feature**: Considers all features simultaneously

### **Example**
```
Sequence 1: MSE = 0.5 (predicted well, normal)
Sequence 2: MSE = 0.8 (predicted well, normal)
Sequence 3: MSE = 5.2 (predicted poorly, ANOMALY)

Threshold = mean(0.5, 0.8, ...) + 3*std(...) = 2.0
Sequence 3: 5.2 > 2.0 → ANOMALY
```

---

## **METHOD 25: Identify Top Contributing Feature (Lines 432-448)**

```python
feature_errors = np.square(y - y_pred)
error_df = pd.DataFrame(feature_errors, columns=features)
anomaly_points['Top_Feature'] = anomaly_points[features].idxmax(axis=1)
```

### **What It Does**
- Calculates error for each feature separately
- Identifies which feature had highest error

### **Why Important**
- **Root cause**: Know WHY it's an anomaly
- **Actionable**: "Error_Rate anomaly" → check service logs
- **Prioritization**: Focus on most impacted features

### **Example**
```
Anomaly at 2025-01-15 14:00:
- Total_Requests error: 0.5
- Error_Count error: 0.3
- Error_Rate error: 4.2 ← Highest
- Request_Rate error: 0.6

Top_Feature = "Error_Rate"
→ Action: Investigate why error rate spiked
```

---

# 🔍 **SECTION 6: UNSUPERVISED LEARNING (35-45 min)**

## **METHOD 26: K-Means Clustering (Lines 505-521)**

```python
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(sequences)
```

### **What It Does**
- Groups similar sequences into 3 clusters
- Each cluster represents a "behavior pattern"

### **Why Important**
- **Pattern discovery**: Automatically finds common behaviors
- **Anomaly detection**: Small clusters = rare patterns = anomalies
- **Behavior profiling**: Cluster 0 = normal, Cluster 1 = peak hours, Cluster 2 = attacks

### **How It Works**
```
Cluster 0 (large): Normal business hours pattern (1000 sequences)
Cluster 1 (medium): Weekend pattern (300 sequences)
Cluster 2 (small): Unusual pattern (10 sequences) → ANOMALIES
```

### **Anomaly Detection Logic**
```python
cluster_counts = np.bincount(cluster_labels)
anomalous_clusters = np.where(cluster_counts < 0.05 * len(cluster_labels))[0]
df['Cluster_Anomaly'] = np.isin(cluster_labels, anomalous_clusters)
```

- Clusters with < 5% of data = anomalies
- Rare patterns are suspicious

---

## **METHOD 27: Dynamic Time Warping (Lines 536-551)**

```python
from fastdtw import fastdtw
distance, path = fastdtw(seq1, seq2, dist=euclidean)
```

### **What It Does**
- Measures similarity between two sequences
- Handles time shifts (e.g., same pattern but 2 hours later)

### **Why Important**
- **Flexible matching**: Recognizes patterns even if timing is slightly off
- **Attack signatures**: Detect similar attack patterns across different times
- **Pattern library**: Build database of known good/bad patterns

### **Example**
```
Sequence 1: [100, 200, 300, 400, 300, 200, 100] (morning spike)
Sequence 2: [100, 100, 200, 300, 400, 300, 200] (same spike, 1 hour delayed)

Euclidean distance: High (different alignment)
DTW distance: Low (same pattern, just shifted)
```

---

## **METHOD 28: Isolation Forest (Lines 617-626)**

```python
iso_forest = IsolationForest(contamination=0.05, random_state=42)
iso_labels = iso_forest.fit_predict(scaled_features)
df['Isolation_Forest_Anomaly'] = iso_labels == -1
```

### **What It Does**
- Isolates outliers by randomly partitioning data
- Anomalies are easier to isolate (fewer splits needed)

### **Why Important**
- **No training needed**: Works without labeled data
- **Fast**: Efficient for high-dimensional data
- **Robust**: Works well with mixed feature types

### **How It Works**
```
Imagine a forest of decision trees:

Normal point (hard to isolate):
- Split 1: Is Total_Requests > 500? Yes → Go right
- Split 2: Is Error_Rate < 10%? Yes → Go right
- Split 3: Is IP_Variability < 100? Yes → Go right
... (many more splits needed)

Anomaly (easy to isolate):
- Split 1: Is Total_Requests > 5000? Yes → ISOLATED!
(Only 1 split needed → anomaly)
```

### **Contamination Parameter**
- `contamination=0.05` means expect 5% of data to be anomalies
- Lower value = stricter detection
- Higher value = more lenient

---

## **METHOD 29: One-Class SVM (Lines 807-812)**

```python
oc_svm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.05)
svm_labels = oc_svm.fit_predict(scaled_features)
df['OCSVM_Anomaly'] = svm_labels == -1
```

### **What It Does**
- Learns a boundary around "normal" data
- Anything outside boundary = anomaly

### **Why Important**
- **Robust**: Works well with complex, non-linear boundaries
- **High-dimensional**: Handles many features simultaneously
- **Complementary**: Catches different anomalies than Isolation Forest

### **How It Works**
```
Imagine a bubble around normal data:

Normal data points: Inside bubble
Anomalies: Outside bubble

The bubble can be any shape (thanks to RBF kernel):
- Circular
- Elliptical
- Irregular

Example:
Point A: (Total_Requests=1000, Error_Rate=5%) → Inside bubble (normal)
Point B: (Total_Requests=5000, Error_Rate=50%) → Outside bubble (anomaly)
```

### **Parameters**
- `kernel='rbf'`: Radial Basis Function (flexible boundary)
- `nu=0.05`: Expected fraction of anomalies (5%)
- `gamma='auto'`: Kernel coefficient (auto-calculated)

---

## **METHOD 30: Explain Unsupervised Anomalies (Lines 673-688)**

```python
feature_contrib = np.abs(scaled_features - scaled_features.mean(axis=0))
top_feature_iso = []
for i, is_anom in enumerate(df['Isolation_Forest_Anomaly']):
    if is_anom:
        idx = np.argmax(feature_contrib[i])
        top_feature_iso.append(features[idx])
```

### **What It Does**
- Calculates deviation from mean for each feature
- Identifies which feature deviated most

### **Why Important**
- **Root cause**: Know which metric caused the anomaly
- **Actionable**: "IP_Variability anomaly" → check for distributed attacks
- **Explainability**: ML models are no longer "black boxes"

### **Example**
```
Anomaly at 2025-01-15 14:00:
Feature deviations from mean:
- Total_Requests: 0.5 std devs
- Error_Count: 0.3 std devs
- IP_Variability: 4.2 std devs ← Highest
- Request_Rate: 0.6 std devs

Top_Feature = "IP_Variability"
→ Explanation: "Anomaly caused by unusual IP diversity (possible distributed attack)"
```

---

# 🎨 **SECTION 7: ENSEMBLE DETECTION (45-50 min)**

## **METHOD 31: Combine All Models (Lines 1125-1130)**

```python
df['Combined_Anomaly'] = (
    df['Isolation_Forest_Anomaly'] |
    df['OCSVM_Anomaly'] |
    df['Prophet_Anomaly'] |
    df['Cluster_Anomaly']
)
```

### **What It Does**
- Combines anomaly flags from all 4 models
- Uses OR logic (flagged by ANY model = anomaly)

### **Why Important**
- **Comprehensive coverage**: Different models catch different anomaly types
- **Reduced false negatives**: Less likely to miss real anomalies
- **Confidence levels**: Anomalies flagged by multiple models are higher priority

### **Example**
```
Timestamp: 2025-01-15 14:00

Isolation_Forest: ✓ Anomaly (high IP_Variability)
OCSVM: ✓ Anomaly (outside normal boundary)
Prophet: ✗ Normal (within forecast range)
Cluster: ✗ Normal (in common cluster)

Combined_Anomaly: ✓ (2 out of 4 models agree)
Confidence: Medium-High
```

---

## **METHOD 32: Determine Anomaly Source (Lines 1170-1183)**

```python
def combined_top_feature(row):
    features = []
    if row.get('Isolation_Forest_Anomaly', False):
        features.append(row['Isolation_Forest_Top_Feature'])
    if row.get('OCSVM_Anomaly', False):
        features.append(row['OCSVM_Top_Feature'])
    if row.get('Prophet_Anomaly', False):
        features.append('Prophet')
    if row.get('Cluster_Anomaly', False):
        features.append('Cluster')
    return ', '.join([f for f in features if f is not None])
```

### **What It Does**
- Lists which models detected the anomaly
- Combines top features from each model

### **Why Important**
- **Confidence indicator**: More models = higher confidence
- **Multi-faceted view**: Different models see different aspects
- **Prioritization**: "All 4 models" > "1 model only"

### **Example Output**
```
Anomaly 1: "Error_Rate, IP_Variability, Prophet"
→ High confidence (3 models), multiple issues

Anomaly 2: "Cluster"
→ Low confidence (1 model), investigate further
```

---

## **METHOD 33: Daily Anomaly Summary (Lines 1223-1224)**

```python
df['Date'] = df['Timestamp'].dt.date
daily_summary = df.groupby('Date')['Combined_Anomaly'].sum().reset_index()
```

### **What It Does**
- Counts total anomalies per day

### **Why Important**
- **Trend detection**: Are anomalies increasing over time?
- **Day comparison**: Which days had most issues?
- **Reporting**: Management wants daily numbers

### **Example**
```
Date         Anomaly_Count
2025-01-13   5
2025-01-14   3
2025-01-15   47  ← Investigate this day!
2025-01-16   6
```

---

## **METHOD 34: Hourly Pattern Analysis (Lines 1227-1230)**

```python
df['Hour'] = df['Timestamp'].dt.hour
hourly_summary = df[df['Combined_Anomaly']].groupby(['Hour'])['Top_Feature_Combined'] \
    .apply(lambda x: Counter(x).most_common(3)) \
    .reset_index()
```

### **What It Does**
- Groups anomalies by hour of day
- Finds top 3 features for each hour

### **Why Important**
- **Recurring patterns**: "Errors spike every day at 2 AM"
- **Root cause**: "2 AM = batch job time → investigate batch jobs"
- **Prevention**: Schedule maintenance during low-anomaly hours

### **Example**
```
Hour  Top_Features
02:00 [('Error_Rate', 15), ('Request_Rate', 3), ('IP_Variability', 1)]
14:00 [('Request_Rate', 20), ('IP_Variability', 10), ('Error_Rate', 2)]
18:00 [('Request_Rate', 25), ('Error_Rate', 5), ('Cluster', 3)]

→ Insight: 2 AM = error issues, 2 PM & 6 PM = traffic issues
```

---

## **METHOD 35: Recurring Pattern Detection (Lines 1241-1242)**

```python
recurring = df[df['Combined_Anomaly']].groupby(['Hour', 'Top_Feature_Combined']).size().reset_index(name='Count')
recurring = recurring[recurring['Count'] > 1]
```

### **What It Does**
- Finds anomalies that occur at same hour multiple times

### **Why Important**
- **Predictable issues**: "Error spike at 2 AM every day"
- **Scheduled fixes**: Can plan maintenance proactively
- **Root cause**: Recurring = likely configuration issue, not attack

### **Example**
```
Hour  Top_Feature     Count
02:00 Error_Rate      7     ← Happens every day!
14:00 Request_Rate    3     ← Happens 3 times this week
18:00 IP_Variability  2     ← Happened twice

→ Action: Fix the 2 AM error issue first (highest frequency)
```

---

## **METHOD 36: Visualize Daily Trends (Lines 1251-1258)**

```python
plt.figure(figsize=(12, 4))
plt.bar(daily_summary['Date'], daily_summary['Anomaly_Count'], color='salmon')
plt.xlabel('Date')
plt.ylabel('Number of Anomalies')
plt.title('Daily Anomaly Trend Across All Models')
```

### **What It Does**
- Creates bar chart of daily anomaly counts

### **Why Important**
- **Visual pattern recognition**: Easier to spot trends than in tables
- **Communication**: Executives understand graphs better than CSV files
- **Anomaly of anomalies**: "Why did Jan 15 have 10x more anomalies?"

---

## **METHOD 37: Visualize Hourly Patterns (Lines 1261-1267)**

```python
plt.figure(figsize=(12, 4))
plt.bar(hourly_summary['Hour'], [sum([c[1] for c in x]) for x in hourly_summary['Top_Features']], color='skyblue')
plt.xlabel('Hour of Day')
plt.ylabel('Anomaly Count')
plt.title('Hourly Anomaly Pattern')
```

### **What It Does**
- Creates bar chart of anomaly counts by hour

### **Why Important**
- **Time-based patterns**: "Most anomalies at 2 AM and 6 PM"
- **Capacity planning**: Scale infrastructure for peak anomaly hours
- **Maintenance scheduling**: Avoid deploying during high-anomaly hours

---

# 🤖 **SECTION 8: AI REPORT GENERATION (50-55 min)**

## **METHOD 38: Load Qwen LLM (Lines 1310-1315)**

```python
from llama_cpp import Llama

llm = Llama.from_pretrained(
    repo_id="Qwen/Qwen2-1.5B-Instruct-GGUF",
    filename="qwen2-1_5b-instruct-fp16.gguf"
)
```

### **What It Does**
- Loads a local Large Language Model (Qwen 2)
- 1.5B parameters, instruction-tuned

### **Why Important**
- **Natural language**: Converts technical data to readable reports
- **Local execution**: No API costs, no data sent to external servers
- **Customizable**: Can be fine-tuned for domain-specific language

---

## **METHOD 39: Generate AI Summary (Lines 1331-1355)**

```python
prompt = f"""
I have the following weekly anomaly trends:

Daily Anomaly Counts:
{daily_text}

Top Features Causing Anomalies:
{features_text}

Generate a concise report explaining:
- Which days had the most anomalies
- Which features caused the most issues
- Any repeating patterns observed
- Important points for root cause analysis
"""

response = llm(prompt, max_tokens=512)
```

### **What It Does**
- Sends data and instructions to LLM
- Receives natural language report

### **Why Important**
- **Management-friendly**: Non-technical stakeholders understand it
- **Actionable**: Includes recommendations, not just facts
- **Time-saving**: Automates report writing

### **Example Output**
```
Weekly Anomaly Report:

Peak Activity: Wednesday (Jan 15) showed the highest anomaly count with 47 incidents, 
significantly above the weekly average of 8 incidents per day.

Critical Features:
- Error_Rate: Responsible for 35% of anomalies, indicating potential service degradation
- Request_Rate: Caused 28% of anomalies, suggesting traffic spikes or DDoS attempts
- IP_Variability: Contributed to 22% of anomalies, pointing to distributed attack patterns

Recurring Patterns:
- Daily error spikes at 2 AM (7 occurrences) - likely related to scheduled batch jobs
- Request rate anomalies cluster around 6 PM (3 occurrences) - peak traffic period

Recommendations:
1. Investigate batch job failures at 2 AM - check logs for specific error messages
2. Scale infrastructure to handle 6 PM traffic surge - consider auto-scaling rules
3. Review security rules for distributed IP patterns - potential bot activity
4. Monitor Wednesday patterns - determine if external factor caused spike
```

---

## **METHOD 40: Generate Actionable Summaries (Lines 1914-1930)**

```python
def generate_actionable_summary(text, section_name):
    prompt = f"""
You are an expert cybersecurity analyst. Summarize the following anomaly data for the section '{section_name}' 
in **4-6 concise sentences**. Make it easy for a non-technical user to understand what is happening. 
Include three parts: 

1. Summary: what the data shows (trends, spikes, anomalies).  
2. Potential causes: what could trigger these anomalies.  
3. Suggested actions: practical mitigation or investigation steps.

Data:
{text}
"""
    response = llm(prompt=prompt, max_tokens=300, temperature=0.3)
    return response['choices'][0]['text'].strip()
```

### **What It Does**
- Generates 3-part summaries: What happened, Why, What to do

### **Why Important**
- **Structured**: Always includes cause and action
- **Actionable**: DevOps team knows exactly what to do
- **Consistent**: Same format for all sections

---

## **METHOD 41: Create PDF Report (Lines 1982-2036)**

```python
pdf_file = 'Actionable_AI_Anomaly_Report.pdf'
doc = SimpleDocTemplate(pdf_file, pagesize=letter)
elements = []

# Add title
elements.append(Paragraph("Actionable AI-Driven Anomaly Detection Report", styles['Title']))

# Add AI summaries
elements.append(Paragraph(daily_summary_ai, styles['BodyText']))
elements.append(Paragraph(top_features_ai, styles['BodyText']))
elements.append(Paragraph(recurring_patterns_ai, styles['BodyText']))

# Add graphs
elements.append(Image(daily_graph, width=500, height=250))
elements.append(Image(top_feature_graph, width=500, height=250))
elements.append(Image(recurring_graph, width=500, height=250))

# Add tables
add_table(daily_summary, "Daily Anomaly Summary Table")
add_table(top_features_unique, "Top Feature Anomalies Table")

doc.build(elements)
```

### **What It Does**
- Creates professional PDF with:
  - Executive summary (AI-generated)
  - Graphs (daily trends, top features, recurring patterns)
  - Tables (detailed data)
  - Recommendations (AI-generated)

### **Why Important**
- **Professional**: Ready to share with management
- **Comprehensive**: All information in one document
- **Automated**: No manual report writing
- **Archival**: Can compare reports week-over-week

---

## **METHOD 42: Enhanced PDF with Graph Summaries (Lines 2047-2218)**

```python
def generate_graph_summary(text, graph_name):
    prompt = f"""
You are a cybersecurity analyst. Summarize the following data for the graph '{graph_name}' 
in 4-6 concise sentences, in plain English for management. Include:
1. What is happening (trends, spikes, anomalies)
2. Possible causes
3. Suggested actions / mitigation steps

Data:
{text}
"""
    response = llm(prompt=prompt, max_tokens=300, temperature=0.3)
    return response['choices'][0]['text'].strip()
```

### **What It Does**
- Generates AI summary for EACH graph
- Explains what the graph shows and why it matters

### **Why Important**
- **Context**: Graphs alone don't explain themselves
- **Insights**: AI finds patterns humans might miss
- **Accessibility**: Non-technical readers understand graphs

---

# 📊 **SECTION 9: COMPLETE WORKFLOW SUMMARY (55-60 min)**

## **End-to-End Pipeline**

```
1. Load Raw Logs (Lines 6-9)
   ↓
2. Aggregate Hourly (Lines 18-23)
   ↓
3. Visualize Baseline (Lines 54-71)
   ↓
4. Prophet Forecasting (Lines 86-151)
   → Detects: Seasonal anomalies, trend deviations
   ↓
5. Feature Engineering (Lines 219-288)
   → Creates: Error_Rate, Request_Rate, Session_Length, etc.
   ↓
6. LSTM Deep Learning (Lines 294-469)
   → Detects: Complex temporal patterns
   ↓
7. K-Means Clustering (Lines 475-522)
   → Detects: Rare behavior patterns
   ↓
8. Isolation Forest (Lines 593-639)
   → Detects: Outliers in high-dimensional space
   ↓
9. One-Class SVM (Lines 804-856)
   → Detects: Points outside normal boundary
   ↓
10. Ensemble Combination (Lines 1125-1210)
    → Combines: All model results
    ↓
11. Trend Analysis (Lines 1213-1284)
    → Identifies: Daily patterns, hourly patterns, recurring issues
    ↓
12. AI Report Generation (Lines 1308-1831)
    → Generates: Natural language summaries
    ↓
13. PDF Creation (Lines 1875-2218)
    → Produces: Professional report with graphs and tables
```

---

## **Key Metrics Tracked**

| Metric | Importance | Anomaly Indicator |
|--------|-----------|-------------------|
| **Total_Requests** | Overall traffic volume | Sudden spikes (DDoS) or drops (outage) |
| **Error_Count** | Service health | Increasing errors = degradation |
| **Error_Rate** | Normalized service health | High % = critical issue |
| **Unique_IPs** | Traffic diversity | High diversity = distributed attack |
| **Unique_Sessions** | User activity | Low sessions + high requests = bots |
| **Request_Rate** | Requests per minute | Sustained high rate = capacity issue |
| **Session_Length** | User behavior | Very short = automated activity |

---

## **Model Strengths & Use Cases**

### **Prophet**
✅ **Best for**: Seasonal patterns, forecasting, trend detection  
✅ **Catches**: Traffic spikes during off-hours, unexpected drops  
❌ **Misses**: Sudden attacks (no history), subtle multi-feature anomalies

### **LSTM**
✅ **Best for**: Complex temporal patterns, multi-feature dependencies  
✅ **Catches**: Unusual sequences, gradual degradation  
❌ **Misses**: Single-point outliers, requires training time

### **Isolation Forest**
✅ **Best for**: Unknown anomalies, high-dimensional data  
✅ **Catches**: Outliers, rare events  
❌ **Misses**: Anomalies within dense clusters

### **One-Class SVM**
✅ **Best for**: Non-linear boundaries, robust outlier detection  
✅ **Catches**: Points far from normal behavior  
❌ **Misses**: Anomalies near decision boundary

### **K-Means Clustering**
✅ **Best for**: Pattern grouping, behavior profiling  
✅ **Catches**: Rare behavior patterns  
❌ **Misses**: Anomalies within large clusters

---

## **Output Files Summary**

| File | Contents | Use Case |
|------|----------|----------|
| `aggregated_logs.csv` | Hourly metrics | Baseline analysis |
| `enhanced_logs.csv` | Engineered features | ML model input |
| `daily_anomaly_summary.csv` | Daily anomaly counts | Trend tracking |
| `top_feature_anomalies.csv` | Most common issues | Prioritization |
| `recurring_anomaly_patterns.csv` | Repeating patterns | Proactive fixes |
| `Anomaly_Report.pdf` | Visual report | Management presentation |
| `Enhanced_AI_Anomaly_Report.pdf` | AI-summarized report | Executive summary |

---

## **Real-World Use Cases**

### **Use Case 1: DDoS Attack Detection**
**Scenario**: Sudden traffic spike from distributed IPs

**Detection Path**:
1. Prophet: Flags traffic above forecast (Line 126)
2. Isolation Forest: Flags high IP_Variability (Line 622)
3. LSTM: Confirms unusual temporal pattern (Line 414)
4. Ensemble: High confidence (3/4 models agree)

**Report Output**:
```
Anomaly: 2025-01-15 14:00
Models: Prophet, Isolation Forest, LSTM
Top Feature: IP_Variability
Confidence: High
Recommendation: Enable rate limiting, investigate IP sources
```

---

### **Use Case 2: Service Degradation**
**Scenario**: Gradual increase in error rate

**Detection Path**:
1. Prophet: Detects upward trend in errors (Line 126)
2. Feature Engineering: High Error_Rate (Line 225)
3. Hourly Analysis: Errors cluster at 2 AM (Line 1227)
4. AI Report: "Recurring service degradation at 2 AM"

**Report Output**:
```
Recurring Pattern: Error_Rate spike at 02:00 (7 occurrences)
Possible Cause: Scheduled batch job failures
Recommendation: Review batch job logs, adjust resource allocation
```

---

### **Use Case 3: Bot Activity**
**Scenario**: Unusual session patterns

**Detection Path**:
1. Feature Engineering: High Session_Per_Request ratio (Line 227)
2. K-Means: Identifies rare pattern cluster (Line 506)
3. One-Class SVM: Confirms abnormal behavior (Line 808)
4. Ensemble: Medium confidence (2/4 models agree)

**Report Output**:
```
Anomaly: Unusual session patterns detected
Top Feature: Session_Per_Request
Possible Cause: Automated bot activity
Recommendation: Enable CAPTCHA, review user agent strings
```

---

## **Tuning Recommendations**

### **High False Positive Rate**
```python
# Increase contamination (allow more anomalies)
iso_forest = IsolationForest(contamination=0.10)  # Up from 0.05

# Widen Prophet confidence interval
forecast = model.predict(future, interval_width=0.99)  # Up from 0.95

# Increase LSTM threshold
threshold = np.mean(mse) + 4 * np.std(mse)  # Up from 3
```

### **Missing Known Anomalies**
```python
# Decrease contamination (stricter detection)
iso_forest = IsolationForest(contamination=0.02)  # Down from 0.05

# Add more features
df['New_Feature'] = df['Metric1'] / df['Metric2']

# Increase LSTM complexity
model.add(LSTM(128, activation='relu'))  # Up from 64
```

### **Slow Performance**
```python
# Reduce LSTM epochs
model.fit(X_train, y_train, epochs=20)  # Down from 50

# Subsample for Isolation Forest
iso_forest = IsolationForest(max_samples=128)  # Down from 256

# Shorter sequences
sequence_length = 5  # Down from 10
```

---

## **Integration Examples**

### **Slack Alerts**
```python
import requests

def send_slack_alert(anomaly_count, top_feature):
    webhook_url = "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
    message = {
        "text": f"🚨 {anomaly_count} anomalies detected! Top issue: {top_feature}",
        "attachments": [{
            "color": "danger",
            "fields": [
                {"title": "Anomaly Count", "value": str(anomaly_count), "short": True},
                {"title": "Top Feature", "value": top_feature, "short": True}
            ]
        }]
    }
    requests.post(webhook_url, json=message)

# Call after ensemble detection
if df['Combined_Anomaly'].sum() > 10:
    send_slack_alert(df['Combined_Anomaly'].sum(), top_features['Feature'].iloc[0])
```

### **Grafana Dashboard**
```python
from prometheus_client import Gauge, push_to_gateway

anomaly_gauge = Gauge('anomaly_count', 'Number of detected anomalies')
anomaly_gauge.set(df['Combined_Anomaly'].sum())

push_to_gateway('localhost:9091', job='anomaly_detection', registry=registry)
```

### **PagerDuty Incident**
```python
def trigger_pagerduty(severity, description):
    if severity == 'high':
        payload = {
            "routing_key": "YOUR_ROUTING_KEY",
            "event_action": "trigger",
            "payload": {
                "summary": description,
                "severity": "critical",
                "source": "AI Anomaly Detection"
            }
        }
        requests.post("https://events.pagerduty.com/v2/enqueue", json=payload)

# Call for high-confidence anomalies
high_confidence = df[(df['Isolation_Forest_Anomaly']) & (df['OCSVM_Anomaly']) & (df['Prophet_Anomaly'])]
if len(high_confidence) > 0:
    trigger_pagerduty('high', f"Multiple models detected {len(high_confidence)} critical anomalies")
```

---

## **Q&A Preparation**

### **Q: Why use multiple models instead of just one?**
A: Each model has different strengths:
- Prophet catches seasonal anomalies
- LSTM catches complex patterns
- Isolation Forest catches outliers
- SVM catches boundary violations
- Clustering catches rare patterns

When multiple models agree, confidence is higher. When only one flags it, requires investigation.

### **Q: How often should we retrain models?**
A: 
- **Prophet**: Weekly (learns new patterns)
- **LSTM**: Monthly (computationally expensive)
- **Isolation Forest**: Daily (fast, adapts quickly)
- **One-Class SVM**: Weekly (moderate cost)
- **K-Means**: Weekly (cluster patterns change)

### **Q: What if we get too many false positives?**
A: Three options:
1. Increase contamination parameter (allow more anomalies)
2. Require multiple models to agree (ensemble voting)
3. Add domain-specific rules (e.g., ignore anomalies during deployment windows)

### **Q: Can this detect zero-day attacks?**
A: Yes, especially with unsupervised methods (Isolation Forest, One-Class SVM). They don't need to "know" what an attack looks like - they just detect "unusual" behavior.

### **Q: How much data is needed to start?**
A: Minimum:
- **Prophet**: 2 weeks (to learn weekly patterns)
- **LSTM**: 1 month (more data = better)
- **Isolation Forest**: 1 week (works with less data)
- **One-Class SVM**: 1 week
- **K-Means**: 1 week

Ideal: 3-6 months for robust models

### **Q: What's the computational cost?**
A:
- **Prophet**: Low (seconds)
- **LSTM**: High (minutes to hours for training)
- **Isolation Forest**: Low (seconds)
- **One-Class SVM**: Medium (minutes)
- **K-Means**: Low (seconds)

Total pipeline: ~10-30 minutes per run (depending on data size)

---

## **Next Steps for Your Team**

1. ✅ **Week 1**: Run script on historical data, validate results
2. ✅ **Week 2**: Tune parameters based on false positive/negative rates
3. ✅ **Week 3**: Integrate with Slack/PagerDuty for real-time alerts
4. ✅ **Week 4**: Set up automated daily runs (cron job or Airflow)
5. ✅ **Week 5**: Create Grafana dashboard for visualization
6. ✅ **Week 6**: Train team on interpreting reports
7. ✅ **Week 7**: Establish incident response procedures
8. ✅ **Week 8**: Review and optimize based on feedback

---

## **Key Takeaways**

1. ✅ **Multi-layered detection** catches more anomalies than single-method
2. ✅ **Feature engineering** is crucial for model performance
3. ✅ **Ensemble approach** reduces false positives
4. ✅ **AI-powered reports** make results accessible to non-technical stakeholders
5. ✅ **Hourly/daily trends** reveal recurring issues for proactive fixes
6. ✅ **Explainability** (top features) enables quick root cause analysis
7. ✅ **Automated pipeline** saves time and ensures consistency

---

**END OF PRESENTATION**

**Questions?**
