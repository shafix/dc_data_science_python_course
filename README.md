# dc_data_science_python_course
Code used when going through the DataCamp "Data Scientist with Python" course





# Matplotlib
## line plot
```
import matplotlib.pyplot as plt
plt.plot(x,y)
plt.show()
```
## scatter plot
```
import matplotlib.pyplot as plt
plt.scatter(x,y) # Optional: s = bubble_size_arr ; c = bubble_color_dict ; alpha = opacity_float
plt.xscale('log') # changes the plot to logarithmic scale
plt.show()
```
## histogram
```
import matplotlib.pyplot as plt
plt.hist(data,bins)
plt.show()
```
## x and y axis labels, plot title, enable grid
```
plt.xlabel('...')
plt.ylabel('...')
plt.title('...')
plt.grid(True)
```
## explicit tick values and labels
```
y_val_arr = [1000, 10000, 100000] # this specific example should probably use plt.xscale('log') as well
y_label_arr = ['1k', '10k', '100k']
plt.yticks(y_val_arr, y_label_arr)
```





# Pandas
## Data frame from dictionary, specify row labels
```
import pandas as pd

# Create dictionary out of lists
names = ['United States', 'Australia', 'Japan', 'India', 'Russia', 'Morocco', 'Egypt']
dr =  [True, False, False, False, True, True, True]
cpc = [809, 731, 588, 18, 200, 70, 45]
my_dict = { 'country': names, 'drives_right': dr, 'cars_per_cap':cpc }

# Builds the dataframe from a dictionary
cars = pd.DataFrame(my_dict) 

# Define row labels / indexes for each row using an array
row_labels = ['US', 'AUS', 'JPN', 'IN', 'RU', 'MOR', 'EG']
cars.index = row_labels
```
## Data frame from csv
```
import pandas as pd
cars = pd.read_csv('cars.csv') # Optional: specify which column should be used as index : index_col = 0 
```
## Getting info about a data frame:
```
df.head() # Prints 1st five rows by default, can provide integer argument for a custom number of 1st rows
df.info() # info about the columns, such as column names and data types
print(df.shape) # Shape of the data frame: (row_nr , col_nr)
df.describe() # summary statistics for each column

df.values # A two-dimensional NumPy array of values.
df.columns # An index of columns: the column names.
df.index # An index for the rows: either row numbers or row names.
```
## Sorting a data frame:
```
df.sort_values(["col_name1","col_name2"]) # optional ascending=[True,False] (for each column by which we are sorting)
```
## Fetching data from the pandas data frame
```
print( cars['country'] ) # returns a "series" object with index(key) + value pairs, labeled 1d array
print( cars[['country']] ) # returns a "data frame" object with the index + single selected column
print( cars[['country','cars_per_cap']] ) # returns a "data frame" object with the index + selected columns

print( cars[1:5] ) # returns a new data frame - slice of the old data frame or certain rows 

print( cars.loc['RU'] ) # returns a row as pandas series - row access location by index/label
print( cars.loc[['RU']] ) # returns a row as pandas data frame - row access location by index/label
print( cars.loc[['RU','US']] ) # returns rows as pandas data frame - row access location by index/label
print( cars.loc[ ['RU','US'] , ['country','cars_per_cap'] ] ) # returns pandas frame of selected rows by index/label + selected columns
print( cars.loc[ : , ['country','cars_per_cap'] ] ) # all rows, certain columns

print( cars.iloc[[1]] ) # returns a row as pandas data frame - row access location by index number
print( cars.iloc[[1,2,3]] ) # returns a row as pandas data frame - row access location by index number
print( cars.iloc[ [1,2,3] , [1,2] ] ) # returns a row as pandas data frame - row access location by index number + column by column position number
print( cars.iloc[ : , [1,2] ] ) # all rows, certain columns
```
## Summary statistics:
```
# Single value result:
df["column"].mean()
df["column"].median()
df["column"].min()
df["column"].max()

# Custom sumarry statistics:
# Example function
def iqr(column):
    return column.quantile(0.75) - column.quantile(0.25)
# Example usage (multiple columns and multiple statistics)
df[['column1','column2']].agg([function1,function2])

# Cumulative (result for each row)
df["column"].cumsum() # cumulative sum
df["column"].cummax() # cumulative max
```
## Duplicate dropping, counting
```
df.drop_duplicates(subset=["col1","col2"]) # distinct on col1, col
df["col"].value_counts() # optional: sort=True ; normalize=True (for proportions)
store_depts["department"].value_counts().sort_values(ascending=False) # another example of sorting by count
store_depts["department"].value_counts().sort_index(ascending=False) # example of sorting by index
print(dept_counts_sorted)
```
## Grouping for aggregate operations, pivot tables
```
df.groupby("col1")["col2"].opperation() # example df.groupby("color")["price"].mean() - average price for each color
df.groupby("col1")["col2"].agg([func1, func2]) # getting multiple aggregates in one call
df.groupby(["col1","col2"])[["col3","col4"]].opperation()
df.groupby("col1")[["col3","col4"]].opperation() # aggregate multiple columns
df.groupby("col1")[["col2","col3"]].opperation() # grouping by several columns
df.groupby(["col1","col2"])[["col3","col4"]].opperation() # group by and aggregate multiple columns

#pivot tables:
dogs.pivot_table( values="weight", index="color", aggfunc=[func1, func2] ) # index=groupby, values=aggregatewhat
#group by 2 columns pivot (gets mean by default), fill NaN with 0:
dogs.pivot_table( values="weight", index="color", columns="breed", fill_value=0, margins=True ) # index=groupby1, columns=groupby2, values=aggregatewhat, fill_value=fillmissingwithsmtng, margins=showsummarystats
```
## Logical operations with numpy arrays
```
np.logical_and(number_arr_1 > 13, number_arr_2 < 15) 
np.logical_or(number_arr_1 > 13, number_arr_2 < 15) 
np.logical_not(number_arr_1 > 13, number_arr_2 < 15) 
```
## Basic control flow (if, elif, else)
```
area = 10.0
if(area < 9) :
    print("small")
elif(area < 12) :
    print("medium")
else :
    print("large")
```
## Filtering (subsetting) pandas data frames
```
cpc = cars['cars_per_cap'] # get a series of numeric values for each country
between = np.logical_and(cpc > 10, cpc < 80) # get a series of boolean values depending on the conditions
medium = cars[between] # select only those countries that have TRUE in the given boolean series

dogs[dogs["height_cm"] > 60]
dogs[dogs["color"] == "tan"]
dogs[(dogs["height_cm"] > 60) & (dogs["color"] == "tan")]

colors = ["brown", "black", "tan"]
condition = dogs["color"].isin(colors)
dogs[condition]
```
## Basic python while and for loops
```
# simple while
x = 1
while x < 4 :
    print(x)
    x = x + 1
    
# array
areas = [1.73, 1.68, 1.71, 1.89] 
for index, value in enumerate(areas) :
    print(str(index) + ': ' + str(value))

# dictionary
world = { "afghanistan":30.55, 
          "albania":2.77,
          "algeria":39.21 }
for key, value in world.items() :
    print(key + " -- " + str(value))
    
# numpy array
np_arr = np.array([2,3,4])
for x in np.nditer(np_arr) :
    ...
    
# pandas data frame
for label, row in brics.iterrows() :
    ...
```
## Loop through pandas data frame and adjust the row
```
# Creates new column and assigns the lenght of another column as value
for lab, row in brics.iterrows() :
    brics.loc[lab, "name_length"] = len(row["country"])

# Same with using apply() - more performant/efficient
brics["name_length"] = brics["country"].apply(len)
```
## Numpy random numbers
```
# Import numpy as np
import numpy as np
import numpy.random as rnd

# Set the seed
rnd.seed(123)

print( rnd.rand() ) # generates a random float between 0 and 1
print( rnd.randint(4, 8) ) # Generates a random integer between 4 and 7 (8 is excluded)
```
## Random number walk exercise
```
# Numpy is imported, seed is set
# Initialize random_walk
random_walk = [0]
for x in range(100) :
    step = random_walk[-1] # Set step: last element in random_walk
    dice = np.random.randint(1,7) # Roll the dice
    # Determine next step
    if dice <= 2:
        step = max(0, step - 1) # using max to make sure it can't go below 0
    elif dice <= 5:
        step = step + 1
    else:
        step = step + np.random.randint(1,7)
    random_walk.append(step) # append next_step to random_walk
print(random_walk)
```

