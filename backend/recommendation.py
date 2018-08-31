
# coding: utf-8

# In[2]:


import numpy as np

size = 20


# In[3]:


sales_order = np.random.randint(10, size = size)
sales_order


# In[4]:


purchase_price = np.random.randint(10, size = size)
purchase_price


# In[12]:


recommend(sales_order, purchase_price, 1, rest=[1,2,3,4,5])


# In[7]:


recommend(sales_order, purchase_price, 2, 0.5)


# In[8]:


recommend(sales_order, purchase_price, 2)


# In[9]:


recommend(sales_order, purchase_price, 3)


# In[20]:


file = open('./purchase_price_prediction.txt', 'r')
purchase_price = file.read().split('\n')[:-1]
purchase_price = [float(i) for i in purchase_price]


# In[24]:


file = open('./sale_quantity_prediction.txt', 'r')
sales_order = file.read().split('\n')[:-1]
sales_order = [float(i) for i in sales_order]


# In[37]:


sum(sales_order)


# In[41]:


file = open('strategy.txt', 'w')
result = recommend(sales_order, purchase_price, array_limit = 15, alpha = 0.9)[1]
for i in result:
    file.write('%d\n' % int(i))
file.flush()
file.close()


# In[35]:


def recommend(sales_order, purchase_price, inventory_period = None, alpha = 1, purchase_days = [0, 1, 2, 3, 4, 5, 6, 7]):
    
    assert len(sales_order) == len(purchase_price)
    size = len(sales_order)
    
    if array_limit is None:
        array_limit = size
    
    array_price = []
    array_index = []
    
    profit = 0
    array_purchase = [0] * size

    for i in range(size):
    
        while (len(array_price) > 0) and (purchase_price[i] <= array_price[-1]):
            array_price.pop()
            array_index.pop()
            
        if (i % 7) in rest: 
            array_price.append(purchase_price[i])
            array_index.append(i)
        
        if (len(array_price) > 0) and (i - array_index[0] >= array_limit):
            array_price.pop(0)
            array_index.pop(0)
        
        history = (int)(sales_order[i] * alpha)
        current = sales_order[i] - history
        
        if len(array_price) > 0:
            profit += array_price[0] * history + array_price[-1] * current
            array_purchase[array_index[0]] += history
            array_purchase[array_index[-1]] += current
        
    return profit, array_purchase

