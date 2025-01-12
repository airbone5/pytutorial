```python
import numpy as np
import matplotlib.pyplot as plt

# Generate data using NumPy
# 100 evenly spaced points between 0 and 10
x = np.linspace(0, 10, 100)  
# Compute sine values for x
y = np.sin(x)  

# Create a line plot
plt.plot(x, y, label='sin(x)', color='blue', linestyle='--')
plt.title('Line Plot of sin(x)')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid(True)
plt.show()
```


    
![png](testconvert.files/testconvert_0_0.png)
    

