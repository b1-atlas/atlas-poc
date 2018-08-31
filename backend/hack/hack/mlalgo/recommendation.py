def recommend(sales_order, purchase_price, inventory_period = None, alpha = 1, purchase_days = [0, 1, 2, 3, 4, 5, 6, 7]):
    
    assert len(sales_order) == len(purchase_price)
    size = len(sales_order)
    
    if inventory_period is None:
        inventory_period = size
    
    array_price = []
    array_index = []
    
    profit = 0
    array_purchase = [0] * size

    for i in range(size):
    
        while (len(array_price) > 0) and (purchase_price[i] <= array_price[-1]):
            array_price.pop()
            array_index.pop()
            
        if (i % 7) in purchase_days: 
            array_price.append(purchase_price[i])
            array_index.append(i)
        
        if (len(array_price) > 0) and (i - array_index[0] >= inventory_period):
            array_price.pop(0)
            array_index.pop(0)
        
        history = (int)(sales_order[i] * alpha)
        current = sales_order[i] - history
        
        if len(array_price) > 0:
            profit += array_price[0] * history + array_price[-1] * current
            array_purchase[array_index[0]] += history
            array_purchase[array_index[-1]] += current
        
    return profit, array_purchase