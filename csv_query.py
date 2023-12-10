import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('fivecarddraw_2M.csv')

# Get the total number of hands in the csv
num_hands = df.shape[0] 

# Query to get the number of hands that improve in class (eg. pair to two pair)
# This is broken down by how dramatic the change in class is
improved_labels = ['>0', '>1', '>2', '>3', '>4', '>5', '>6', '>7', '>8']
improved_counts = []
for dif in range(9):
    improved_hands = df[(df['class'] > (df['s_class'] + dif))]
    improved_counts.append(improved_hands.shape[0])


# Use Seaborn for additional styling
sns.set(style="whitegrid", font_scale=1.2)

# Create the graph for the improved hands query
plt.figure(figsize=(10, 6))
bar_colors = sns.color_palette("viridis", len(improved_labels))
plt.bar(improved_labels, improved_counts, color=bar_colors)

# Add labels above each bar
for i, count in enumerate(improved_counts):
    plt.text(i, count + 0.1, str(count), ha='center', va='bottom')
plt.title(f'Hands Improved by Swapping (Total Hands = {num_hands})')
#plt.suptitle(f'Total Hands: {num_hands}', x=0.5, y=0.92, fontsize=14, fontweight='bold')
plt.ylabel('Number of Hands')
plt.xlabel('Minimum Hand Rank Improvement')
plt.yticks(fontweight='bold')
plt.xticks(fontweight='bold')
# Adjust num hands to be logged
plt.yscale('log')
plt.show()

# Create a bar graph that shows the number of hand in each class before and after swapping
# The x-axis should be the classes (0-9) and the y-axis should be the total number of hands in that class
# The two bars are total from df['class'] and df['s_class']
# Define hand classes
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
#df = pd.read_csv('fivecarddraw.csv')

# Define hand classes
hand_classes = ['High Card', 'Pair', 'Two Pair', 'Three of a Kind', 'Straight', 'Flush', 'Full House', 'Four of a Kind', 'Straight Flush', 'Royal Flush']

# Create lists to store counts for each class before and after swapping
initial_classes = [df[df['class'] == i].shape[0] for i in range(9, -1, -1)]
swapped_classes = [df[df['s_class'] == i].shape[0] for i in range(9, -1, -1)]

# Use Seaborn for additional styling
sns.set(style="whitegrid", font_scale=1.2)

# Create a bar graph
plt.figure(figsize=(12, 6))
bar_width = 0.35
bar_positions = range(10)

# Plot for before swapping
for i, count in enumerate(initial_classes):
    plt.text(i, count + 0.1, str(count), ha='center', va='bottom', rotation=45, fontsize=8)

# Plot for after swapping
for i, count in enumerate(swapped_classes):
    plt.text(i + bar_width, count + 0.1, str(count), ha='center', va='bottom', rotation=45, fontsize=8)

plt.bar(bar_positions, initial_classes, width=bar_width, label='Before Swapping', alpha=0.7, color='purple')
plt.bar([pos + bar_width for pos in bar_positions], swapped_classes, width=bar_width, label='After Swapping', alpha=0.7, color='orange')

plt.xlabel('Hand Class')
plt.ylabel('Number of Hands (log scale)')
plt.title('Number of Hands in Each Class Before and After Swapping')
plt.xticks([pos + bar_width / 2 for pos in bar_positions], hand_classes, rotation=45, ha='right')
plt.yscale('log')  # Set the y-axis to a logarithmic scale
plt.legend()
plt.tight_layout()
plt.show()
