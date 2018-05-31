# https://matplotlib.org/gallery/misc/table_demo.html

import numpy as np
import matplotlib.pyplot as plt


data = [[ 75.27, 75.00, 58.38, 58.25, 62.5, 74.78, 76.37, 77.15],
        [ 58.33, 63.75, 72.50, 75, 73.75, 72.91, 75.41, 74.54],
        [ 65.73, 68.92, 64.68, 65.57, 67.68, 73.83, 75.89, 75.84]]
        
columns = ('USDA', 'Twitter', 'WordNet',
            'Pluralization', 'Phrase Machine', 'Banned Words',
            'Span Merging', 'WSD')
rows = ['%s year' % x for x in ['Precision', 'Recall', 'F1-Score']]

values = np.arange(0, 100, 1)
value_increment = 1

# Get some pastel shades for the colors
colors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows)))
n_rows = len(data)

index = np.arange(len(columns)) + 0.3
bar_width = 0.4

# Initialize the vertical-offset for the stacked bar chart.
y_offset = np.zeros(len(columns))

# Plot bars and create text labels for the table
cell_text = []
for row in range(n_rows):
    plt.bar(index, data[row], bar_width, bottom=y_offset, color=colors[row])
    y_offset = y_offset + data[row]
    cell_text.append(['%1.1f' % (x / 1.0) for x in y_offset])
# Reverse colors and text labels to display the last value at the top.
colors = colors[::-1]
cell_text.reverse()

# Add a table at the bottom of the axes
the_table = plt.table(cellText=cell_text,
                      rowLabels=rows,
                      rowColours=colors,
                      colLabels=columns,
                      loc='bottom')

# Adjust layout to make room for the table:
plt.subplots_adjust(left=0.2, bottom=0.2)

plt.ylabel("Loss in ${0}'s".format(value_increment))
plt.yticks(values * value_increment, ['%d' % val for val in values])
plt.xticks([])
plt.title('Loss by Disaster')

plt.show()