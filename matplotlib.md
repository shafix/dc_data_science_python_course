# Matplotlib
## line plot
```
import matplotlib.pyplot as plt
plt.plot(x,y)

or

df.plot(x="date",y="weight",kind="line") # rot=45 - rotates x axis label by 45 degrees
```
## scatter plot
```
import matplotlib.pyplot as plt
plt.scatter(x,y) # Optional: s = bubble_size_arr ; c = bubble_color_dict ; alpha = opacity_float
plt.xscale('log') # changes the plot to logarithmic scale

or 

df.plot(kind="scatter", x="height", y="weight")
```
## histogram
```
import matplotlib.pyplot as plt
plt.hist(data,bins)

or 

df["height"].hist(bins=5)

# Layering 2 histograms
dogs[dogs["sex"]=="F"]["height"].hist(alpha=0.7)
dogs[dogs["sex"]=="M"]["height"].hist(alpha=0.7)
plt.legend(["F","M"])
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
## barplot
```
avg_weight_by_breed=dog_pack.groupby("breed")["weight_kg"].mean()
avg_weight_by_breed.plot(kind="bar", title="...")
```
## pyplot interface
```
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
# figure - object, container, holds everything that we see on the page
# axis - object, part of the page that holds the data
```
## Line plot
```
ax.plot(
	DS_COL1, # ['Mon','Tue']
	DS_COL2, # [5, 12]
	marker="o", # Marker style
	linestyle="--", # Line style, use none to remove line
	color="r" # red
)

ax.plot( seattle_weather["MONTH"], seattle_weather["MONTHLY_AVERAGE_TEMP"] ) # Line plot by default
ax.plot( texas_weather["MONTH"], texas_weather["MONTHLY_AVERAGE_TEMP"] ) # Add another plot on the same axis, both will be shown
plt.show()
```
## Axis label setting , tittle setting
```
ax.set_xlabel("bla bla")
ax.set_ylabel("bla bla")
ax.set_title("bla bla")
```
## Small multiples / subplots
```
fig, ax = plt.subplots(3, 2) # creates a figure with 3 rows of subplots and 2 columns
print(ax.shape) # [3,2]
ax[0,0].plot(...)
fig, ax = plt.subplots(2, 1) # creates a figure with 2 rows
print(ax.shape) # [2]
ax[1].plot(...)
fig, ax = plt.subplots(2, 1, sharey=True) # Makes both of the subplots share the same Y values
```


## Timeseries!
```
# Import CSV with date field as index:
climate_change = pd.read_csv("climate_change.csv", parse_dates=["date"], index_col="date")

# Plot using the date index:
ax.plot( climate_change.index, climate_change["relative_temp"] )

# Zoom in using date index:
seventies = climate_change["1970-01-01":"1979-12-31"]
ax.plot(seventies.index, seventies["co2"])
```
## Using twin axes:
```
fig, ax = plt.subplots()
ax.plot(..., color='blue') # Ex: y data meassured in the 1000s
ax.set_ylabel(..., color='blue')
ax.tick_params('y', color='blue')
ax2 = ax.twinx() # creates the 2nd y axes
ax2.plot(..., color='red') # Ex: y data meassured in the 10s
ax2.set_ylabel(..., color='red')
ax.tick_params('y', color='red')
plt.show()
```
## Custom function for plotting several time series on top of each other with different y scales and colors
```
def plot_timeseries( axes, x, y, color, xlabel, ylabel ):
	axes.plot(x, y, color=color)
	axes.set_xlabel(xlabel)
	axes.set_ylabel(ylabel, color=color)
	axes.tick_params('y', colors=color)
# Usage
fig, ax = plt.subplots()
plot_timeseries(ax, XDATA, YDATA1, 'blue', XLABEL, YLABEL1)
ax2 = ax.twinx() # creates the 2nd y axes
plot_timeseries(ax2, XDATA, YDATA2, 'red', XLABEL, YLABEL1)
plt.show()
```
## Annotating part of the plot
```
ax.annotate(
	">1 degree", 
	xy=[pd.Timestamp("2015-10-06"), 1], 
	xytext=(pd.Timestamp("2008-10-06"), -0.2),
	arrowprops={"arrowstyle":"->", "color":"gray"} 
)
```
## Barcharts - Quantitative comparisons
```
df=pd.read_csv('xxx.csv', index_col=0) # use 1st column as index when importing csv
fig, ax = plt.subplots() # create figure and axes
ax.bar(medals.index, medals["Gold"], label="Gold") # create bar with X (categories) and Y (numbers) values suplied
ax.bar(medals.index, medals["Silver"], bottom=medals["Gold"], label="Silver") # stack anoter bar on top of the previous one
ax.bar(medals.index, medals["Bronze"], bottom=medals["Gold"] + medals["Silver"], label="Bronze") # ... and another one!
ax.set_xticklabels(medals.index, rotation=90) # use X (categories) as X axes labels and rotate them by 90degrees
ax.set_ylabel("Number of medals") # Set Y label
ax.legend() # Add legend
plw.show()
```
## Histograms - Quantitative distributions
```
fig, ax = plt.subplots()
ax.hist(mens_rowing["Height"], label="Rowing", bins=5, histtype="step") # can specify bin limits bins=[12,22,55,66]...
ax.hist(mens_gymnastics["Height"], label="Gymnastics", bins=5, histtype="step")
ax.set_xlabel("Height in CM")
ax.set_ylabel("# of observations")
ax.legend()
plt.show()
```
## Statistical plotting : error bars 
```
# Bar plot with means + standard deviations as error bars
fig, ax = plt.subplots()
ax.bar(	"Rowing",
		mens_rowing["Height"].mean(),
		yerr=mens_rowing["Height"].std() )
ax.bar(	"Gymnastics",
		mens_gymnastics["Height"].mean(),
		yerr=mens_gymnastics["Height"].std() )
ax.set_ylabel("Height (cm)")
plt.show()


# Line chart with error lines
fig, ax = plt.subplots()
ax.errorbar( seattle_weather["MONTH"], seattle_weather["MLY-TAVG-NORMAL"], yerr=seattle_weather["MLY-TAVG-STDDEV"] )
ax.errorbar( austin_weather["MONTH"],  austin_weather["MLY-TAVG-NORMAL"],  yerr=austin_weather["MLY-TAVG-STDDEV"]  )
ax.set_ylabel("Temperature (Fahrenheit)")
plt.show()
```

## Boxplot
```
fig, ax = plt.subplots()
ax.boxplot( [ mens_rowing["Height"], mens_gymnastics["Height"] ] ) 
ax.set_xticklabels(["Rowing","Gymnastics"])
ax.set_ylabel("Height (cm)")
plt.show()
```
# Scatterplot
```
fig, ax = plt.subplots()
# Note: can use the attribute "c=..." to pass,for example, a time/date column to color the dots according to the time (creates a nice gradient)
ax.scatter(climate_change1["co2"], climate_change1["relative_temp"], color="red", label="cc1")
ax.scatter(climate_change2["co2"], climate_change2["relative_temp"], color="blue", label="cc2")
ax.set_xlabel("co2")
ax.set_ylabel("temp")
ax.legend()
plt.show()
```
## Changing the style of the figure
```
plt.style.use("ggplot")
plt.style.use("default")
```
## Saving the figure to a file
```
fig.savefig("filename.png") # can use quality=X [1-100] - compression ; can use dpi=X [300] - resolution ; png, svg, jpg - file format
```
## Set the size of the figure
```
fig.set_size_inches([3,5])
```
## Automating figure creation
```
fig, ax = plt.subplots()
sports=summer_2016_medals["Sport"].unique() # returns an array/series of unique sport categories
for sport in sports:
	sport_df = summer_2016_medals[summer_2016_medals["Sport"] == sport ]
	ax.bar( sport, sport_df["Height"].mean(), yerr=sport_df["Height"].std() )
ax.set_ylabel("Height (cm)") # Set Y label
ax.set_xticklabels(sports, rotation=90)
plt.show()
```
