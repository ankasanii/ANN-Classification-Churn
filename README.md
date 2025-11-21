# exprements.ipynb Line-by-Line Reference (With Usage Notes)

Each cell is reproduced with two layers of commentary:

- A descriptive comment explaining the line.
- An added `Use:` comment clarifying why the line exists / when you would apply it.

You can copy any block into a Python environment and run it as-is. Comments are safe to keep or remove.

---

## Cell 1: Imports

```python
# Import pandas for tabular data handling
import pandas as pd  # Use: Core library for DataFrame operations (loading, cleaning, transforming).
# Split arrays or matrices into random train and test subsets
from sklearn.model_selection import train_test_split  # Use: Create unbiased train/test partitions.
# Standardize features and encode categorical labels
from sklearn.preprocessing import StandardScaler, LabelEncoder  # Use: Scale numeric features & convert string categories to numbers.
# Serialize (save/load) Python objects (encoders, scaler, etc.)
import pickle  # Use: Persist preprocessing artifacts for later inference.
```

## Cell 2: Load Dataset

```python
# Marker comment indicating start of dataset loading
## Load dataset  # Use: Segments notebook logically for readability.
# Read the churn modelling CSV file into a DataFrame
data = pd.read_csv('Churn_Modelling.csv')  # Use: Load raw customer churn data into memory for processing.
# Display the DataFrame in notebook (last expression)
data  # Use: Quick visual inspection / sanity check of loaded data.
```

## Cell 3: Drop Unnecessary Columns

```python
# Marker comment for preprocessing start
## Data Preprocessing  # Use: Denotes beginning of feature-cleaning phase.
# Explain intent: remove columns not useful for modeling
# Drop unnecessary identifier columns
# RowNumber, CustomerId, Surname are removed
data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)  # Use: Remove columns that don't help prediction (avoid noise / leakage).
# Show modified DataFrame
data  # Use: Verify columns were dropped correctly.
```

## Cell 4: Label Encode Gender

```python
# Marker comment for categorical encoding
## Encoding categorical variables  # Use: Start transforming text categories to model-friendly formats.
# Create a label encoder instance for the 'Gender' column
label_encoder_gender = LabelEncoder()  # Use: Prepare object to map string labels to integers.
# Fit encoder on Gender values and replace with numeric codes
data['Gender'] = label_encoder_gender.fit_transform(data['Gender'])  # Use: Store numeric representation for faster model consumption.
# Display DataFrame after encoding
data  # Use: Confirm encoding applied correctly.
```

## Cell 5: One-Hot Encode Geography

```python
# Marker comment for one-hot encoding 'Geography'
## One-hot encoding for 'Geography'  # Use: Prepare for turning Geography into separate binary features.
# Import OneHotEncoder from sklearn
from sklearn.preprocessing import OneHotEncoder  # Use: Provides one-hot encoding capability.
# Instantiate encoder (default sparse output)
onehot_encoder_geo = OneHotEncoder()  # Use: Object will learn unique geography categories.
# Fit encoder on Geography column and transform to sparse matrix
geography_encoded = onehot_encoder_geo.fit_transform(data[['Geography']])  # Use: Convert categorical geography to machine-readable format.
# Display sparse matrix object
geography_encoded  # Use: Inspect raw encoded structure (memory-efficient form).
```

## Cell 6: Geography Feature Names

```python
# Retrieve generated one-hot feature names for 'Geography'
onehot_encoder_geo.get_feature_names_out(['Geography'])  # Use: Get column names to assign meaningful headers to encoded DataFrame.
```

## Cell 7: Convert Sparse to DataFrame

```python
# Convert sparse matrix to dense array and wrap in DataFrame with columns
geography_encoded_df = pd.DataFrame(geography_encoded.toarray(), columns=onehot_encoder_geo.get_feature_names_out(['Geography']))  # Use: Create explicit columns for integration into main dataset.
# Display encoded geography DataFrame
geography_encoded_df  # Use: Verify one-hot encoding expansion.
```

## Cell 8: Merge Encoded Columns

```python
# Marker to indicate merging encoded columns into original data
## combine encoded columns with original data  # Use: Replace raw geography with structured binary features.
# Drop original Geography column and concatenate new one-hot columns
data = pd.concat([data.drop('Geography', axis=1), geography_encoded_df], axis=1)  # Use: Ensure model sees numeric-only feature set.
# Show updated DataFrame containing one-hot geography
data  # Use: Check new feature columns present.
```

## Cell 9: Save Encoders

```python
# Marker for persistence of encoders
## save the encoders and scaler  # Use: Persist objects to avoid refitting during inference/deployment.
# Save the gender label encoder
with open('label_encoder_gender.pkl', 'wb') as file:  # Use: Open file for binary write.
    pickle.dump(label_encoder_gender, file)  # Use: Store trained label encoder.
# Save the geography one-hot encoder
with open('onehot_encoder_geo.pkl', 'wb') as file:  # Use: Open file for binary write.
    pickle.dump(onehot_encoder_geo, file)  # Use: Store trained one-hot encoder.
```

## Cell 10: Split and Scale Data

```python
# Marker for feature/target separation
## divide the dataset into independent and dependent variables  # Use: Begin modeling phase by defining X and y.
# Features matrix excluding target label
X = data.drop('Exited', axis=1)  # Use: All inputs the model will learn from.
# Target vector (churn indicator)
y = data['Exited']  # Use: Output labels (0/1 churn).

# Marker for train/test split
## Split the dataset into training and testing sets  # Use: Create evaluation holdout to measure generalization.
# 80% train, 20% test, reproducible split via random_state
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Use: Avoid overfitting by early separation.

# Marker for scaling features
## scale the features  # Use: Normalize feature scales to stabilize training.
# Instantiate standard scaler
scaler = StandardScaler()  # Use: Object stores mean & std of training data.
# Fit scaler on training features and transform
X_train = scaler.fit_transform(X_train)  # Use: Learn scaling parameters + apply to training set.
# Transform test features using training stats
X_test = scaler.transform(X_test)  # Use: Apply identical scaling (no leakage).
```

## Cell 11: Save Scaler

```python
# Persist the fitted scaler for inference
with open('scaler.pkl', 'wb') as file:  # Use: Open destination for scaler persistence.
    pickle.dump(scaler, file)  # Use: Ensures identical transformation during future predictions.
```

## Cell 12: TensorFlow / Keras Imports

```python
# Import TensorFlow core API
import tensorflow as tf  # Use: Deep learning framework backend.
# Sequential model class
from tensorflow.keras.models import Sequential  # Use: Allows stacking layers linearly.
# Dense (fully connected) layer
from tensorflow.keras.layers import Dense  # Use: Core layer type for ANN.
# Training callbacks (EarlyStopping, TensorBoard)
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard  # Use: Add training control & logging instrumentation.
# General keras alias
from tensorflow import keras  # Use: Access high-level APIs conveniently.
# Import layers namespace for convenience
from tensorflow.keras import layers  # Use: Shorthand for referencing layer classes.
# Datetime for timestamped logging directories
import datetime  # Use: Generate unique log folder names.
# Empty string (no operational effect)
""  # Use: Harmless artifact; can be removed.
```

## Cell 13: Build ANN Model

```python
# Marker for model construction
## Build the ANN model  # Use: Start defining network architecture.
# Define sequential feed-forward network
model = keras.Sequential([
    # First hidden layer: 64 ReLU units, input shape matches number of features
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  ## Hidden layer connected to input layer  # Use: Extract nonlinear interactions among features.
    # Second hidden layer: 32 ReLU units
    layers.Dense(32, activation='relu'),  ## Hidden layer 2  # Use: Learn higher-level representations.
    # Output layer: single sigmoid unit for binary classification probability
    layers.Dense(1, activation='sigmoid')  ## Output layer  # Use: Produce probability of churn (0–1).
])
# Print model architecture summary
model.summary()  # Use: Inspect parameter counts and layer shapes.
```

## Cell 14: Optimizer and Loss Objects

```python
# (Redundant re-import of tensorflow; harmless)
import tensorflow  # Use: Not required again; kept without issue.
# Instantiate Adam optimizer with specified learning rate
opt = tensorflow.keras.optimizers.Adam(learning_rate=0.01)  # Use: Controls weight updates (higher LR speeds training, may degrade stability).
# Create binary cross-entropy loss object
loss = tensorflow.keras.losses.BinaryCrossentropy()  # Use: Measures divergence between predicted probability and true label.
# Display the loss object representation
loss  # Use: Visual confirmation of loss instance.
```

## Cell 15: Compile Model

```python
# Marker for compilation
## compile the model  # Use: Finalize training configuration before fitting.
# Configure model with optimizer, loss, and accuracy metric
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])  # Use: Binds optimizer & metrics to model for training loop.
```

## Cell 16: TensorBoard Setup

```python
# Marker for TensorBoard configuration
## setup TensorBoard  # Use: Enable rich training visualization in browser / notebook.
# Build log directory path with current timestamp
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # Use: Ensure unique run separation.
# Create TensorBoard callback; histogram_freq=1 logs histograms every epoch
tensorflow_callbacks = TensorBoard(log_dir=log_dir, histogram_freq=1)  # Use: Track losses, metrics, weights.
```

## Cell 17: EarlyStopping Setup

```python
# Marker for early stopping configuration
## set up EarlyStopping  # Use: Prevent overfitting & unnecessary epochs.
# Stop training if validation loss doesn't improve for 5 epochs, restore best weights
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)  # Use: Keeps model at best validation performance.
```

## Cell 18: Train Model

```python
# Marker for training block
## Train the model  # Use: Initiate learning process.
# Fit model for up to 100 epochs with validation, using callbacks for logging and early stopping
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, callbacks=[early_stopping, tensorflow_callbacks])  # Use: Produces training history object for analysis/plotting.
```

## Cell 19: Save Trained Model

```python
# Marker for model persistence
## save the trained model  # Use: Store final model for reuse/deployment.
# Save model architecture and weights to HDF5 file
model.save('ann_churn_model.h5')  # Use: Enables later loading without retraining.
```

## Cell 20: Load TensorBoard Extension

```python
# Marker for loading notebook extension
## load Tensorboard Extension  # Use: Activate TensorBoard magic commands in notebook environment.
# Jupyter magic to load TensorBoard extension
%load_ext tensorboard  # Use: Prepares inline TensorBoard UI.
```

## Cell 21: Launch TensorBoard

```python
# Launch TensorBoard pointing to logs directory
%tensorboard --logdir logs/fit  # Use: View training curves and metrics interactively.
```

## Cell 22: Reload Artifacts for Inference

```python
# Marker for prediction phase
## Predetermine.  # Use: Transition from training to inference.
## Load the trained model, scaler pickle, onehot, encoder pickle files  # Use: Restore preprocessing + model state.
# Import model loader
from tensorflow.keras.models import load_model  # Use: Function to load saved .h5 model.
# Import pickle (redundant but safe)
import pickle  # Use: Deserialize Python objects.
# Load saved ANN model
model = load_model('ann_churn_model.h5')  # Use: Ready model for predictions.
# Load fitted scaler
with open('scaler.pkl', 'rb') as file:  # Use: Access scaling parameters.
    scaler = pickle.load(file)  # Use: Reapply same scaling to new data.
# Load one-hot encoder for Geography
with open('onehot_encoder_geo.pkl', 'rb') as file:  # Use: Access category mapping for geography.
    onehot = pickle.load(file)  # Use: Ensure consistent one-hot columns.
# Load label encoder for Gender
with open('label_encoder_gender.pkl', 'rb') as file:  # Use: Ensure same mapping Male/Female -> int.
    label_encoder = pickle.load(file)  # Use: Prepare categorical transformation.
# Two spaces (formatting artifact)

# Empty line (no effect)
""  # Use: Harmless; can be removed.
```

## Cell 23: Example Input Dictionary

```python
# Marker for example input construction
# Example input data for prediction  # Use: Provide test case for inference.
# Define a single synthetic customer record for inference
input_data = {
    'CreditScore': 600,  # Use: Numerical feature (creditworthiness).
    'Geography': 'France',  # Use: Categorical feature (will be one-hot encoded).
    'Gender': 'Male',  # Use: Categorical (label encoded).
    'Age': 40,   # Use: Numerical demographic feature.
    'Tenure': 3,  # Use: Duration with bank (relationship length).
    'Balance': 60000,  # Use: Account balance size.
    'NumOfProducts': 2,  # Use: Cross-sell indicator.
    'HasCrCard': 1,  # Use: Binary credit card ownership.
    'IsActiveMember': 1,  # Use: Engagement signal.
    'EstimatedSalary': 50000  # Use: Income feature.
}
# Single space string (artifact)
" "  # Use: Not needed; can be removed.
```

## Cell 24: One-Hot Encode Geography for Input

```python
# Marker for geography encoding of input
# one hot encode 'Geography'  # Use: Prepare input to match training feature space.
# Transform geography value to one-hot vector
geo_encoded = onehot.transform([[input_data['Geography']]]).toarray()  # Use: Convert single category to multi-column binary vector.
# Wrap encoded vector in DataFrame with correct column names
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot.get_feature_names_out(['Geography']))  # Use: Align column names with training set.
# Display encoded geography row
geo_encoded_df  # Use: Inspect transformed geography representation.
```

## Cell 25: Convert Input Dict to DataFrame

```python
# Convert dict to single-row DataFrame
input_df = pd.DataFrame([input_data])  # Use: Standardize input structure for concatenation operations.
# Show raw input DataFrame
input_df  # Use: Visual validation before transformations.
```

## Cell 26: Merge One-Hot Geography

```python
# Concatenate encoded geography columns with remaining input features
input_data_final = pd.concat([input_df.drop('Geography', axis=1), geo_encoded_df], axis=1)  # Use: Replace raw category with encoded binary features.
# Display combined feature row before gender encoding
input_data_final  # Use: Confirm structure prior to label encoding.
```

## Cell 27: Encode Gender

```python
# Marker for encoding categorical variables
## Encode categorical variables  # Use: Ensure input matches numeric-only model expectations.
# Apply previously fitted gender label encoder
input_data_final['Gender'] = label_encoder.transform(input_data_final['Gender'])  # Use: Preserve same mapping used during training.
# Display updated feature row
input_data_final  # Use: Verify categorical transformation success.
```

## Cell 28: Scale Input Features

```python
# Marker for scaling input
# scale the input data  # Use: Apply identical normalization used in training.
# Transform features using saved scaler
input_data_scaled = scaler.transform(input_data_final)  # Use: Prevent feature magnitude bias in prediction.
# Display scaled numpy array
input_data_scaled  # Use: Inspect final numeric tensor.
```

## Cell 29: Predict Churn Probability

```python
# Marker for prediction
## predict the churn probability  # Use: Execute inference pass.
# Generate churn probability (sigmoid output)
churn_probability = model.predict(input_data_scaled)  # Use: Returns probability of churn for provided input.
# Show raw prediction array
churn_probability  # Use: Examine raw numeric output.
```

## Cell 30: Interpret Prediction

```python
# Print formatted churn probability percentage
print(f"Churn Probability: {churn_probability[0][0]*100:.2f}%")  # Use: Human-readable probability display.
# Blank line for readability
""  # Use: Output spacing (cosmetic).
# Decision threshold at 0.5
if churn_probability[0][0] > 0.5:  # Use: Simple rule to convert probability to class label.
    # Likely churn classification
    print("The customer is likely to churn.")  # Use: Actionable interpretation for >50% probability.
else:
    # Not likely churn classification
    print("The customer is not likely to churn.")  # Use: Negative class interpretation.
```

---

## Optional Usage Notes

1. Ensure all required packages in `requirements.txt` are installed.
2. Maintain column order consistency when deploying inference code.
3. Consider wrapping preprocessing + model into a single pipeline for production.

## Quick Re-run Sequence

To re-run training quickly (assuming CSV present):

```python
# 1. Load data
# 2. Preprocess (drop, encode, scale)
# 3. Build & compile model
# 4. Fit & save artifacts
# 5. Predict with new input
```

End of reference document.
