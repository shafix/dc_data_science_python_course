# Writing functions

### Docstrings
Main styles: google style and numpydoc style
```
# Checking the docstring of the function
some_func.__doc__
inspect.getdoc(some_func)
```
Note: DRY and "Do one thing"

### Context managers
Example:
```
with ...(args) as ...:
	...
	...
...
```
Writing one:
```
@contextlib.contextmanager
def my_context():
	# Set up code, does something before returning(yealding)
	yield 99
	# Teardown/cleanup code, does something after returning(yealding)
```
Nested context manager example:
```
# Open both files
with open(src, 'r') as f_src:
	with open(dst, 'w') as f_dst:
		# Read and write each line, one at a time
		for line in f_src:
			f_dst.write(line)
```
Handling errors with try, except, finally:
```
def get_printer(ip):
	p = connect_to_printer(ip)
	try:
		yield
	finally: # Whether the actual usage of the printer succeeds or fails, the printer still gets disconnected this way
		p.disconnect()
		print('disconnected from printer')
		
with get_printer('...') as printer:
	printer.print_page(doc['txt']) <- Throws some kind of error
```

### Functions as objects
```
# Assign function to a variable
def my_function():
	print('Hello')
x = my_function
type(x) # function
x() # Hello

# Functions as list members and usage
list_of_functions = [my_function, open, print]
list_of_functions[2]('Print me!') # Print me!

# Functions as dict members and usage
dict_of_functions = { 'func1': my_function, 'func2': open, 'func3': print }
dict_of_functions['func3']('Print me!') # Print me!
```
### Scopes, global / nonlocal scope / closures
```
# Glocal scope example
x = 7
def foo():
	global x
	x = 42
	print(x) # 42
print(x) # 42! (not 7), because of the "global x" line
# Also "nonlocal x" can be used in case of nested functions

# Closures - nonlocal (parent function in nested functions) variables attached to a returned function
def parent(arg_1, arg2):
	# These 2 variables will be passed along with the child function when the function is returned
	value = 22
	my_dict = {'chocolate':'yummy'}
	
	def child():
		print(2 * value)
		print(my_dict['chocolate'])
		print(arg_1 + arg_2)
	
	return child

new_function = parent(3, 4) # child function + parent function scope variables + given arguments

print([cell.cell_contents for cell in new_function.__closure__]) # [3, 4, 22, {'chocolate':'yummy'}]
```
### Decorators:
Decorators wrap around functions. They can: 
modify the input before passing it to the function, 
modify the output of the function 
modify the behaviour of the function
```
# Decorator - multiplies arguments pass to the original function by 2	
def double_args(func):
	def wrapper(a, b):
		# Call the passed in function, but double each argument
		return func(a * 2, b * 2)
	return wrapper

# Original function
multiply(a, b):
	return a * b

# Using the double_args wrapper to create a new function or modify the multiply function
new_multiply = double_args(multiply)
new_multiply(1,5) # 20 (1*2 * 5*2)
# or... 
multiply = double_args(multiply)
multiply(1,5) # 20
# or..
@double_args
multiply(a, b):
	return a * b
multiply(1,5) # 20
```
Example with args, kwargs:
```
def timer(func):
	"""
	A decorator that prints how long a function took to run.
	"""
	def wrapper(*args, **kwargs):
		t_start = time.time()
		result = func(*args, **kwargs)
		t_total = time.time() - t_start
		print('{} took {}s'.format(func.__name__, t_total))
		return result
	return wrapper
```
Using wraps from functools to preserve original docstrings and other function metadata when creating the wrapper
```
from functools import wraps
...
@wraps(func)
def wrapper(*args, **kwargs):
	...
...
some_decorated_function.__wrapped__ # This actually returns the original undecorated function
```
Decorator factories or decorators that take arguments:
```
def run_n_times(n):
	def decorators(func):
		def wrapper(*args, **kwargs):
			for i in range(n):
				func(*args, **kwargs)
		return wrapper
	return decorator

@run_n_times(3)
def print_sum(a,b):
	print(a+b)
	
def timeout(n_seconds):
	def decorator(func):
		@wraps(func)
		def wrapper(*args, **kwargs):
			# Set an alarm for n seconds
			signal.alarm(n_seconds)
			try:
				# Call the decorated func
				return func(*args, **kwargs)
			finally:
				# Cancel alarm
				signal.alarm(0)
		return wrapper
	return decorator

@timeout(5)
def foo():
	time.sleep(10)
	print('foo!')
```
