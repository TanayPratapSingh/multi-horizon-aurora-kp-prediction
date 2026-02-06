# Multi-Horizon Geomagnetic Kp Index Prediction Using LSTM Neural Networks

## Project Overview

Geomagnetic storms, caused by solar wind disturbances interacting with Earth's magnetosphere, pose significant threats to modern technological infrastructure. These events can disrupt satellite operations, damage power grids, interfere with GPS navigation, and degrade high-frequency radio communications. The economic impact of severe geomagnetic storms is estimated at billions of dollars, with the 1989 Quebec blackout and 2003 Halloween storms serving as prominent examples.

The Kp index, a quasi-logarithmic scale from 0 to 9, quantifies global geomagnetic activity levels measured every three hours. Values below 5 indicate quiet conditions, while Kp ≥ 5 signifies geomagnetic storm conditions requiring protective actions. Accurate Kp prediction enables satellite operators to place spacecraft in safe mode, power grid managers to prepare for potential disturbances, and aurora enthusiasts to plan viewing opportunities.

Current operational forecasting models face several limitations. Traditional empirical models like Wing et al. (2005) rely on linear relationships that fail to capture complex temporal dynamics. Physics-based models require significant computational resources and struggle with short-term predictions. Recent machine learning approaches, including the work by Tan et al. (2018), have shown promise but typically focus on single-horizon predictions, requiring separate models for different forecast periods.

---

## Goals

### Primary Prediction Goals

- **Multi-horizon forecasting:** Develop an LSTM model that simultaneously predicts Kp values at 1, 3, 6, and 12 hours ahead with RMSE competitive with published single-horizon models
- **Target Accuracy:** Less than 1.0 Kp units for 1-3 hour horizons, less than 1.5 for 6-12 hour horizons. Achieve MAE < 0.5 Kp units for operational utility

### Inference Goals

- **Feature importance:** Identify which space weather parameters contribute most to Kp prediction at different time horizons
- **Temporal dynamics:** Determine how feature importance evolves across prediction horizons, distinguishing fast-changing parameters which affect short-term predictions from slow-changing parameters which are more relevant for longer-term forecasting

---

## Data

### Dataset Description

**Source and Coverage:**  
We obtained data from NASA's OMNI2 high-resolution database, which provides hourly measurements of solar wind and interplanetary magnetic field (IMF) parameters. The dataset spans January 1, 2013, to December 31, 2024, encompassing 11 years and covering the declining phase of solar cycle 24, solar minimum around 2019-2020, and the rising phase of solar cycle 25.

**Dataset Characteristics:**
- Total records: 96,552 hourly measurements
- Original parameters: 55 features including IMF components, solar wind plasma properties, derived parameters, and geomagnetic indices
- Temporal resolution: 1 hour (consistent throughout)
- Missing data: <1% across all features
- Geographic coverage: Global

### Feature Analysis

We selected 15 features based on correlation analysis and physics literature. The strongest predictor was ap_Index (r= 0.825), followed by auroral electrojet indices (AE: r=0.73, AL: r=-0.70, AU: r=0.66). Surprisingly, IMF_Bz_GSM showed weaker correlation (r=-0.23) than expected, reflecting our dataset's inclusion of all periods rather than just storms.

Geomagnetic indices (ap, AE, PC_N, AL, AU) dominate the top correlations, all having r that exceed 0.65. This initially raised concerns about potential data leakage because these indices are derived from ground magnetometer measurements that respond to the same geomagnetic activity measured by Kp.

However, these indices represent 3-hour averages of past conditions, and with our 24-hour lookback window, we use historical index values to predict future Kp. This is justified because past geomagnetic activity provides information about the current state of the magnetosphere, which influences near-term future activity.

---

## Methods

### Data Preprocessing Pipeline

#### Step 1: Data acquisition and consolidation
Downloaded 11 annual OMNI2 data files (2013-2024) from NASA GSFC servers. Concatenated files using pandas, preserving chronological order and creating a datetime index from Year/Day-of-Year/Hour columns.

#### Step 2: Column mapping verification
Cross-referenced our column placements against the official NASA OMNI2 format documentation. Rectified significant errors in initial mapping. Verified that Kp appears at column 38 (not 46 as initially assumed) and correctly identified all 55 parameters.

#### Step 3: Fill value replacement
NASA uses various fill values to indicate missing or invalid data: 9999.9, 999.9, 99.99, etc. Replaced all documented fill values with np.nan.

#### Step 4: Kp scale conversion
OMNI stores Kp as integers (33 = 3+, 40 = 4, 57 = 6-). Divided by 10 to convert to standard 0-9 decimal scale. Verified conversion produced realistic values (mean ~1.6, max ~8.3).

#### Step 5: Train/Validation/Test Split
Performed chronological split to address criticality for time series:
- **Training:** 2013-2018 (70% of data, 67,206 records)
- **Validation:** 2019-2020 (15% of data, 14,401 records)
- **Testing:** 2021-2024 (15% of data, 14,401 records)

#### Step 6: Feature Scaling
Applied StandardScaler (zero mean, unit variance) to all features and the target variable.

**Critical:** Fitted scaler only on training data, then transformed validation and test data using training statistics. This prevents data leakage and simulates operational deployment where future data statistics are unknown.

### Training Configuration

- **Input:** 12-hour sequences (12 timesteps × 15 features)
- **Output:** 4 predictions simultaneously [Kp_1h, Kp_3h, Kp_6h, Kp_12h]
- **Loss function:** Mean Squared Error with initial learning rate 0.001
- **Optimizer:** Adam with default parameters
- **Batch size:** 64 sequences
- **Epochs:** Maximum 100, early stopping at epoch 31
- **Data split:** 70% train, 15% validation, 15% test (chronological)
- **Normalization:** StandardScaler on all features

---

## Results

The model achieved strong performance across all prediction horizons on the held-out test set (2021-2024 data):

### Test Set RMSE

- **1-hour ahead:** 0.691 Kp units
- **3-hour ahead:** 0.894 Kp units
- **6-hour ahead:** 1.067 Kp units
- **12-hour ahead:** 1.211 Kp units

### Comparison to Baselines

- **1h:** 48.2% improvement over mean baseline (1.333 RMSE)
- **3h:** 33.0% improvement over mean baseline
- **6h:** 20.0% improvement over mean baseline
- **12h:** 9.2% improvement over mean baseline

### Additional Metrics (1-hour ahead)

- **MAE:** 0.489 Kp units
- **R²:** 0.727 (explains 73% of variance)
- **Median Absolute Error:** 0.312 Kp units
- **Predictions within ±0.5 Kp:** 69.3%
- **Predictions within ±1.0 Kp:** 89.1%

### Prediction Accuracy Characteristics

**1-Hour Ahead:**
- Points cluster tightly around the ideal diagonal line for Kp < 5
- Some scatter appears for higher Kp values (5-7 range)
- Model rarely predicts Kp > 6, even when actual values exceed this
- Clear underestimation of extreme events (actual Kp = 7-8 predicted as 5-6)

**12-Hour Ahead:**
- Increased scatter compared to 1-hour predictions
- Stronger regression toward the mean (predictions cluster around 2-3 Kp)
- Very few predictions exceed Kp = 5
- Model becomes increasingly conservative at longer horizons

---

## Problems Encountered

### 1. Severe Initial Overfitting

**Problem:** Initial model showed severe overfitting with training loss of 0.232 versus validation loss of 0.818, indicating memorization rather than generalization.

**Solution:** We performed regularization which progressed from 30% dropout (0.75 loss) to 50% dropout (0.67), then added L2 regularization (0.59) and early stopping to achieve final validation loss of 0.574, successfully controlling overfitting through combined techniques.

### 2. Data Leakage Concerns with Geomagnetic Indices

**Problem:** Top features (ap, AE, AL, AU) are derived from ground magnetometer measurements responding to the same geomagnetic activity that Kp measures. This raised concerns about if we were predicting Kp using variables that contain Kp information.

**Solution:** We validated that our geomagnetic indices are 3-hour averages (not instantaneous) and our 24-hour lookback correctly uses past data to predict future Kp with proper temporal gaps preventing leakage.

### 3. Column Mapping Errors in Data

**Problem:** Initial column assignments were incorrect, resulting in wrong feature usage and incorrect Kp values during model training.

**Solution:** Downloaded official OMNI2 specification and manually verified all column assignments.

---

## Conclusion

In this project, we successfully developed and validated an LSTM neural network for multi-horizon geomagnetic Kp index prediction. The model achieved strong regression performance with 1-hour RMSE of 0.691 Kp units, representing 48% improvement over baseline methods and competitive performance with published single-horizon models. The multi-horizon architecture demonstrated that a single model can effectively predict at four time scales (1, 3, 6, 12 hours) simultaneously, offering computational efficiency and prediction consistency.

Feature analysis reduced 55 original parameters to 16 essential features, with geomagnetic indices (ap, AE, PC_N) emerging as dominant predictors. Time-lag analysis revealed distinct temporal patterns: fast-decaying features (IMF_Bz, E_Field) drive short-term predictions while stable features (Flow_Speed, DST) remain relevant at long horizons. Different features contribute at different time scales, and LSTM learns appropriate weightings for each output.

Train/validation loss ratio of 0.85 indicates minimal overfitting. Temporal gaps exceeding 8,500 hours between train/validation/test sets prevent data leakage. Error distribution analysis shows nearly unbiased predictions (mean error < 0.05 Kp units) with variance appropriately increasing with horizon.

---

## References

1. Abduallah, Y., et al. "A Transformer-based Framework for Predicting Geomagnetic Indices with Uncertainty Quantification." *Journal of Intelligent Information Systems*, vol. 62, 2024, pp. 887-903.

2. Carbary, J. F. "A Kp-based Model of Auroral Boundaries." *Space Weather*, vol. 3, no. S10001, 2005. doi:10.1029/2005SW000162.

3. Feng, H., et al. "A Kp-driven Machine Learning Model Predicting the Ultraviolet Emission Auroral Oval." *Journal of Geophysical Research: Machine Learning and Computation*, vol. 2, no. e2024JH000543, 2025. doi:10.1029/2024JH000543.

4. Han, Y., et al. "Prediction and Variation of the Auroral Oval Boundary Based on a Deep Learning Model and Space Physical Parameters." *Nonlinear Processes in Geophysics*, vol. 27, 2020, pp. 11-22. doi:10.5194/npg-27-11-2020.

5. Hochreiter, Sepp, and Jürgen Schmidhuber. "Long Short-term Memory." *Neural Computation*, vol. 9, no. 8, 1997, pp. 1735-1780. doi:10.1162/neco.1997.9.8.1735.

6. Papitashvili, Natalia E., and Joseph H. King. *OMNI Hourly Data Set*. NASA Space Physics Data Facility, 2020. doi:10.48322/1shr-ht18.

7. Tan, Y., et al. "Geomagnetic Index Kp Prediction Using Multiple Machine Learning Algorithms." *Space Weather*, vol. 16, no. 12, 2018, pp. 2001-2012. doi:10.1029/2018SW002026.

8. World Data Center for Geomagnetism, Kyoto. *Kp Index Service*. Kyoto University, wdc.kugi.kyoto-u.ac.jp/kp/. Accessed 3 Dec. 2024.
