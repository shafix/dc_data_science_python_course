# dc_data_science_python_course
Code used when going through the DataCamp "Data Scientist with Python" course

# matplotlib
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
