#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd

# Load the dataset (replace 'your_file.csv' with your file's path)
df = pd.read_csv('../TextFiles/moviereviews2.tsv', sep='\t')

# Assuming the dataset has a 'text' column and a 'label' column
review = df['review'].values
labels = df['label'].values


# In[3]:


df.dropna(inplace=True)
print(df.isnull().sum())


# In[4]:


# Assuming the dataset has a 'review' column and a 'label' column
texts = df['review'].values
labels = df['label'].values


# In[6]:


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Define the tokenizer
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")  # Use 10,000 most common words and an OOV token
tokenizer.fit_on_texts(texts)

# Convert the text to sequences of integers
sequences = tokenizer.texts_to_sequences(texts)

# Pad the sequences to ensure uniform input shape
padded_sequences = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')  # maxlen adjusts sequence length


# In[7]:


from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Encode labels
label_encoder = LabelEncoder()
numeric_labels = label_encoder.fit_transform(labels)

# Convert to one-hot encoding (optional for classification)
categorical_labels = to_categorical(numeric_labels)


# In[8]:


from sklearn.model_selection import train_test_split

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(padded_sequences, categorical_labels, test_size=0.2, random_state=42)


# In[9]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# Define the model
model = Sequential([
    Embedding(input_dim=10000, output_dim=64, input_length=100),  # Adjust input_dim and output_dim as needed
    LSTM(64, return_sequences=False),  # Use an LSTM layer
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(categorical_labels.shape[1], activation='softmax')  # Number of classes in categorical_labels
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()


# In[40]:


# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=150,
    batch_size=32,
    verbose=1
)


# In[41]:


# Evaluate on validation data
loss, accuracy = model.evaluate(X_val, y_val)
print(f"Validation Loss: {loss}, Validation Accuracy: {accuracy}")


# In[42]:


# Save the model
model.save('sentiment_analysis_model_2.h5')

# Save the tokenizer
import pickle
with open('tokenizer.pkl', 'wb') as file:
    pickle.dump(tokenizer, file)


# In[45]:


#from tensorflow.keras.models import load_model

# Load the model
saved_model = load_model('sentiment_analysis_model_2.h5')


# In[49]:


loss, accuracy = saved_model.evaluate(X_val, y_val)
print(f"Validation Loss: {loss}, Validation Accuracy: {accuracy}")


# In[53]:


# Example input sentence
new_sentence = "This movie was fantastic! I loved it."
#new_sentence = "This movie was complete and utter garbage. Avoid!!!"

# Tokenize and pad the new sentence
new_sequence = tokenizer.texts_to_sequences([new_sentence])  # Convert text to sequence
new_padded = pad_sequences(new_sequence, maxlen=100, padding="post", truncating="post")  # Pad sequence

# Predict sentiment
prediction = model.predict(new_padded)

# Interpret the result
if prediction[0][0] > 0.5:
    print("Negative Sentiment")
else:
    print("Positive Sentiment")

    


# In[55]:


# Predict on the test set
predictions = model.predict(X_val)

# Convert probabilities to binary predictions
predicted_labels = (predictions > 0.5).astype(int)

# Calculate accuracy
from sklearn.metrics import accuracy_score
manual_accuracy = accuracy_score(y_val, predicted_labels)

print(f"Manual Accuracy: {manual_accuracy:.2%}")


#accuracy = accuracy_score(y_val, predicted_labels)

#print(f"Accuracy: {accuracy}")



# In[31]:


from sklearn.metrics import classification_report

# Print classification report
print(classification_report(y_val, predicted_labels, target_names=["Negative", "Positive"]))


# In[ ]:




