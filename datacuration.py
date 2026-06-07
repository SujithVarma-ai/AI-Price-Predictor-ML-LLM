import os
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import random
from pricer.items import Item
from pricer.parser import parse
from pricer.loaders import ItemLoader


load_dotenv(override=True)

hf_token = os.environ.get("HF_TOKEN")
print("HF_TOKEN =", repr(hf_token))


def main():
# Dataset Loading
# We are loading only Appliances category from the dataset.
    dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_meta_Appliances", split="full", trust_remote_code=True)
    print(f"Number of Appliances: {len(dataset):,}")
    print(dataset[6])



# You will see a pricer folder, in that you can see items.py
# items.py is nothing but there are 94,327 in that dataset(appliances), in that dataset valid are 35,307 items and invalid are 59,020 items. We are going to use only valid items for training our model. So we will parse the dataset and get the valid items and store them in a list called items. You can see the code below.
# Parsing is converting raw text to structured format. In our case, we are converting the raw text of the dataset to structured format of Item class which is defined in items.py. The parse function is defined in parser.py which takes raw text and category as input and returns an Item object if the parsing is successful, otherwise it returns None. We will use tqdm to show the progress of parsing the dataset.
    items = [parse(datapoint, "Appliances") for datapoint in tqdm(dataset)]
    items = [item for item in items if item is not None]
    print(f"There are {len(items):,} items from {len(dataset):,} datapoints")

    print(items[0])
    print(items[0].full)

    prices = [item.price for item in items]
    lengths = [len(item.full) for item in items]

# Plot the distribution of lengths
    plt.figure(figsize=(15, 6))
    plt.title(f"Lengths: Avg {sum(lengths)/len(lengths):,.0f} and highest {max(lengths):,}\n")
    plt.xlabel('Length (chars)')
    plt.ylabel('Count')
    plt.hist(lengths, rwidth=0.7, color="lightblue", bins=range(0, 6000, 100))
    plt.show()

# Get the item with the longest description in the dataset and prints full description of that item.
    max_length = max(lengths)
    max_length_item = items[lengths.index(max_length)]
    print(max_length_item.full)

# Plot the distribution of prices
    plt.figure(figsize=(15, 6))
    plt.title(f"Prices: Avg {sum(prices)/len(prices):,.2f} and highest {max(prices):,}\n")
    plt.xlabel('Price ($)')
    plt.ylabel('Count')
    plt.hist(prices, rwidth=0.7, color="orange", bins=range(0, 1000, 10))
    plt.show()



# These are other categories in the dataset, you can load them as well if you want to train your model on more data. You can use the same code as above to load other categories as well.
# You will see a pricer folder, in that you can see loaders.py
    loader = ItemLoader("Appliances") # We have done for home applianesces category, you can do for other categories as well.
    items = loader.load()
# If they ask for only Appliances category, we can stop it here.


    dataset_names = [
        "Automotive",
        "Electronics",
        "Office_Products",
        "Tools_and_Home_Improvement",
        "Cell_Phones_and_Accessories",
        "Toys_and_Games",
        "Appliances",
        "Musical_Instruments",
    ]

    items = []
    for dataset_name in dataset_names:
        loader = ItemLoader(dataset_name)
        items.extend(loader.load())

    print(f"A grand total of {len(items):,} items") # 293327 items in total from all categories.



    print(items[1000]) 


# We have 2933577 items in total. So these all are valid items. In these items there are duplicates so we will remove them.
    random.seed(42)
    random.shuffle(items)
    seen = set()
    items = [x for x in tqdm(items) if not (x.title in seen or seen.add(x.title))]
    seen = set()
    items = [x for x in tqdm(items) if not (x.full in seen or seen.add(x.full))]
    del seen
    print(f"After deduplication, we have {len(items):,} items") 
# After deduplication, we have 2,887,890 items.


# Plot the distribution of lengths after deduplication
    lengths = [len(item.full) for item in items]
    plt.figure(figsize=(15, 6))
    plt.title(f"Text length: Avg {sum(lengths)/len(lengths):,.1f} and highest {max(lengths):,}\n")
    plt.xlabel('Length (characters)')
    plt.ylabel('Count')
    plt.hist(lengths, rwidth=0.7, color="skyblue", bins=range(0, 4050, 50))
    plt.show()

# Plot the distribution of prices
    prices = [item.price for item in items]
    plt.figure(figsize=(15, 6))
    plt.title(f"Prices: Avg {sum(prices)/len(prices):,.1f} and highest {max(prices):,}\n")
    plt.xlabel('Price ($)')
    plt.ylabel('Count')
    plt.hist(prices, rwidth=0.7, color="blueviolet", bins=range(0, 1000, 10))
    plt.show() 
 

# It shows all the categories in the dataset and how many items are there in each category. 
# It is a bar plot where x-axis is the categories and y-axis is the count of items in each category.
    from collections import Counter
    category_counts = Counter([item.category for item in items])

    categories = category_counts.keys()
    counts = [category_counts[category] for category in categories]

    plt.figure(figsize=(15, 6))
    plt.bar(categories, counts, color="goldenrod")
    plt.title('How many in each category')
    plt.xlabel('Categories')
    plt.ylabel('Count')
    plt.xticks(rotation=30, ha='right')

    for i, v in enumerate(counts):
       plt.text(i, v, f"{v:,}", ha='center', va='bottom')

    plt.show()


# training sample, select 820,000 items from full dataset.
    np.random.seed(42)
    print(f"Items available: {len(items):,}")
    SIZE = min(820_000, len(items))
    prices = np.array([it.price for it in items], dtype=float)
    categories = np.array([it.category for it in items])
    p = (prices - prices.min()) / (prices.max() - prices.min() + 1e-9)

    w = p**2
    w[categories == "Tools_and_Home_Improvement"] *= 0.5
    w[categories == "Automotive"] *= 0.05

    w = w / w.sum()
    idx = np.random.choice(len(items), size=SIZE, replace=False, p=w)
    sample = [items[i] for i in idx]


# Histogram
    prices = [item.price for item in sample]
    plt.figure(figsize=(15, 6))
    plt.title(f"Prices: Avg {sum(prices)/len(prices):,.1f} lowest {min(prices):,} and highest {max(prices):,}\n")
    plt.xlabel('Price ($)')
    plt.ylabel('Count')
    plt.hist(prices, rwidth=0.7, color="blueviolet", bins=range(0, 1000, 10))
    plt.show()


# Bar chart
    from collections import Counter
    category_counts = Counter([item.category for item in sample])
    categories = category_counts.keys()
    counts = [category_counts[category] for category in categories]
    plt.figure(figsize=(15, 6))
    plt.bar(categories, counts, color="goldenrod")
    plt.title('How many in each category')
    plt.xlabel('Categories')
    plt.ylabel('Count')
    plt.xticks(rotation=30, ha='right')
# Add value labels on top of each bar
    for i, v in enumerate(counts):
       plt.text(i, v, f"{v:,}", ha='center', va='bottom')
# Display the chart
    plt.show()


# Pie chart
# Automotive still in the lead, but improved somewhat
# For another perspective, let's look at a pie
    plt.figure(figsize=(12, 10))
    plt.pie(counts, labels=categories, autopct='%1.0f%%', startangle=90)
# Add a circle at the center to create a donut chart (optional)
    centre_circle = plt.Circle((0,0), 0.70, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    plt.title('Categories')
# Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')  
    plt.show()


# Scatter plot 
# How does the price vary with the character count?
    sizes = [len(item.full) for item in sample]
    prices = [item.price for item in sample]
# Create the scatter plot
    plt.figure(figsize=(15, 8))
    plt.scatter(sizes, prices, s=0.2, color="red")
# Add labels and title
    plt.xlabel('Size')
    plt.ylabel('Price')
    plt.title('Is there a simple correlation with text length?')
# Display the plot
    plt.show()

# Scatter plot
# How does the price vary with the weight?
    ounces = [item.weight for item in sample]
    prices = [item.price for item in sample]
# Create the scatter plot
    plt.figure(figsize=(15, 8))
    plt.scatter(ounces, prices, s=0.2, color="darkorange")
# Add labels and title
    plt.xlabel('Weight (ounces)')
    plt.ylabel('Price')
    plt.xlim(0, 400)
    plt.title('Is there a simple correlation with weight?')
# Display the plot
    plt.show()


# items_raw_full and items_raw_lite are the names of the datasets that we are going to push to Hugging Face Hub. 
# items_raw_full is the full dataset with 820,000 items 
# items_raw_lite is the lite dataset with 22,000 items.
# We are pushing these datasets to Hugging Face Hub so that we can use them later for training our model.

    username = "SujithVarma2005"
    full = f"{username}/items_raw_full"
    lite = f"{username}/items_raw_lite"

# Data is divied into 3 types - training data, validation data and test data. 
# Training data is used to train the model, 
# Validation data is used to evaluate its performance
# Test data is used to evaluate the final model. 
# We are using 800,000 items for training, 10,000 items for validation and 10,000 items for testing. 
# We are pushing these datasets to Hugging Face Hub so that we can use them later for training our model.
    train = sample[:800_000]
    val = sample[800_000:810_000]
    test = sample[810_000:]
# Total are 8,20,000

    Item.push_to_hub(full, train, val, test)

    train_lite = train[:20_000]
    val_lite = val[:1_000]
    test_lite = test[:1_000]
#  Total are 22,000

    Item.push_to_hub(lite, train_lite, val_lite, test_lite)

if __name__ == "__main__":
    main()