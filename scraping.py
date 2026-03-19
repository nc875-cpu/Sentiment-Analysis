import pandas as pd
import numpy as np
import random

# 1. Setup the scale
num_rows = 1000000 

# 2. Define the 'DNA' of your iPhone 17 dataset
locations = ['New York, USA', 'London, UK', 'Tokyo, Japan', 'Mumbai, India', 'Sydney, Australia', 'Berlin, Germany', 'Dubai, UAE', 'Paris, France', 'Seoul, South Korea', 'Toronto, Canada']
models = ['iPhone 17', 'iPhone 17 Air', 'iPhone 17 Pro', 'iPhone 17 Pro Max']
sentiments = ['Positive', 'Neutral', 'Negative']

# Templates to ensure diversity
positive_templates = ["Love the new {model} design!", "The A19 chip is a beast.", "Best camera in {location}!", "The battery on {model} lasts forever."]
negative_templates = ["Too expensive.", "The {model} overheating is real.", "Not much better than the 16.", "Charging speed in {location} is slow."]
neutral_templates = ["Just got it today.", "Testing the {model} now.", "Standard Apple update.", "Wait for the sale in {location}."]

# 3. Generate the data efficiently
data = []
for i in range(num_rows):
    sentiment = np.random.choice(sentiments)
    location = np.random.choice(locations)
    model = np.random.choice(models)
    
    if sentiment == 'Positive':
        comment = np.random.choice(positive_templates).format(model=model, location=location)
    elif sentiment == 'Negative':
        comment = np.random.choice(negative_templates).format(model=model, location=location)
    else:
        comment = np.random.choice(neutral_templates).format(model=model, location=location)
        
    data.append([i, location, model, comment, sentiment])

# 4. Create the DataFrame and Save
df = pd.DataFrame(data, columns=['ID', 'Location', 'Model', 'Comment', 'Sentiment'])

# Export as CSV (Recommended over Excel for 1M rows)
df.to_csv('iphone17_1M_dataset.csv', index=False)
print("Dataset successfully created!")