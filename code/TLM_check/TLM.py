def Xn(X,a):
    import numpy as np
    
    return(np.roll(X,-a + 1)[0])

def Fx(y):
    import numpy as np
    Fx = np.zeros((40,40))
    for row in range(40):
        L1 = np.zeros(40)
        
        a = row
        
        L1[0] = -Xn(y,a)
        L1[1] = Xn(y,a + 2) - Xn(y, a - 1)
        L1[2] = -1
        L1[3] = Xn(y,a)
        
        Fx[row,:] = np.roll(L1,row + -2)
    
    return(Fx)