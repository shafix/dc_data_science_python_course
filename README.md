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
## Fetching data from the data frame
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


