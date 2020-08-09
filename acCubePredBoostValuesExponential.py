import numpy as np 
import csv
import math
from scipy.optimize import curve_fit

#Input data. Our normal X value is 2
X = [0,1,2,3,4,5] #This is the different amount of boosts we could give it

#Output data
y = [0, 0.3, 2.7, 5.77, 11.7, 17.4] #This is the cooresponding amount off the ground your character is as a result of the boost. NOTE: WE HAVE FLOATS AS OUTPUTS, SO WE NEED TO CONVERT THEM



#Some testing data (lets see if it can predict the future)




#Make them numpy arrays
X = np.array(X)
y = np.array(y)

#Normal quadratic equation. ax^2 + bx + c
def func(x, a, b, c):
	return ((a * (x ** 2)) + (b * x) + c)


#The function, X and y, p0 which should be kept as such. It's just the range for value testing for X
popt, pcov = curve_fit(func, X, y, p0=[1, 1e-6, 1]) #Setup p0 like this. Without this, it asks for C. If that doesn't work, p0=[5,0.1]


a = popt[0].round(2)
b = popt[1].round(2)  #Get the variables (perfect)
c = popt[2].round(2)



#We're modeling our normal equation here with the predicted X and Y values
def modeledEquation(x):
	global a 
	global b
	global c #Specify that these are defined elsewhere
	return ((a * (x ** 2)) + (b * x) + c)



#WE TAKE OUR EQUATION HERE AND SOLVE FOR X. WE GOT THE Y VALUE AND THE A,B, AND C VALUES 
#https://cdn.discordapp.com/attachments/690575246943715401/742125942943645879/SolvedForX.PNG
"""
1. We have all the variables
https://cdn.discordapp.com/attachments/690575246943715401/742130910597480468/allVariables.PNG


2. We solve for X. Modeled both equations below
https://cdn.discordapp.com/attachments/690575246943715401/742125942943645879/SolvedForX.PNG

"""
######################################################
def predInputValuePlus(yValue):
	global a 
	global b 
	global c
	return (-b + (math.sqrt( (b**2) - ((4*a) * (c - yValue)) )) / (2 * a) )



#This is the other X value solution
def predInputValueMinus(yValue):
	global a 
	global b 
	global c
	return (-b - (math.sqrt((b**2) - ((4*a) * (c - yValue)) )) / (2 * a))

######################################################





#Calculate R^2 value. This is using the training data
######################################################
residuals = y - func(X, *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y-np.mean(y)) ** 2)
rSquaredTrainingData = 1 - (ss_res / ss_tot)
print(rSquaredTrainingData)
######################################################



######################################################
normalJumpHeight = 2.7 #This is the normal Y value (jump height) for the player
jmpBoost = 2 #This is the jump boost

yValHeight = normalJumpHeight * 2; #We want to know the input value for a jump height of the normal height*2

#There's two X values since it's a quadratic equation, but we're only interested in the positive value, so get the max
predXVal = max(predInputValuePlus(yValHeight), predInputValueMinus(yValHeight));
######################################################


print("The Y value is your HEIGHT RESULTING FROM A BOOST. The X value is THE BOOST AMOUNT\n")
print(f"The predicted X (boost) value for the inputted Y value (your height), which we're inputting your normal height times the jump boost  ({normalJumpHeight} * {jmpBoost} = {yValHeight}, so just inputting the y value of {yValHeight}) is {predXVal.round(2)}\n")
print(f"Y value: {yValHeight}. Outputted X value: {predXVal}")


