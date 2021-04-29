
import cv2
from matplotlib import pyplot as plt
import numpy as np

# to get the color of the array sigment 0 black 1 pink 2 wight

def imgColor(image):
    pink = 0
    wight = 0
    for r in range(200):
        for c in range(200):
            h, s, v = image[r][c]
            if (h == 150 and s == 150 and v == 200):
                pink = pink + 1
            if (h == 255 and s == 255 and v == 255):
                wight = wight + 1

    if (pink > wight):
        if(pink>500):
            value=1
        else:
            value=0
    else:
        if (wight > 500):
            value = 2
        else:
            value = 0


    return value




#img1
img1=cv2.imread("x.png")
black=cv2.imread("black.png")
img1=cv2.resize(img1,(600,500))
img1=cv2.cvtColor(img1,cv2.COLOR_BGR2HSV)
black=cv2.resize(black,(600,600))
black[100:600,0:600]=img1
img1=black
#img2
img2=cv2.imread("y.jpg")
img2=cv2.resize(img2,(600,600))
img2=cv2.cvtColor(img2,cv2.COLOR_BGR2HSV)



#simplifying
for i in range(600):
    for j in range(600):
        h,s,v=img1[i,j]
        if(h in range(90,120) and s in range(30,60) and v in range(190,254)):
            img1[i][j]=255,255,255
        elif (h in range(150, 165) and s in range(90, 140) and v in range(160, 200)):
            img1[i][j]=150,150,200
        else:
            img1[i][j]=0,0,0

for i in range(600):
    for j in range(600):
        h,s,v=img2[i,j]
        if(h in range(90,120) and s in range(30,60) and v in range(190,254)):#wight
            img2[i][j]=255,255,255
        elif (h in range(150, 175) and s in range(60, 100) and v in range(100, 180)):#pink
            img2[i][j]=150,150,200
        else:
            img2[i][j]=0,0,0

cv2.imshow("img2",img2)
cv2.waitKey()
#arraying

arr1=[[img1]*3]*3
for r in range(0,img1.shape[0],200):
    for c in range(0,img1.shape[1],200):
         arr1[int(r / 200)][int (c/200)]=img1[r:r+200, c:c+200]
         #cv2.imwrite(f"img{r}_{c}.png",img1[r:r+200, c:c+200,:])

arr2=[[img1]*3]*3
for r in range(0, img2.shape[0], 200):
    for c in range(0, img2.shape[1], 200):
        arr2[int(r / 200)][int(c / 200)] = img2[r:r + 200, c:c + 200]


#the color of ech sigment in the array
arr1Value=[[0]*3]*3
arr2Value=[[0]*3]*3

for r in range(3):
    for c in range(3):
        arr1Value[r][c]=imgColor(arr1[r][c])

for r in range(3):
    for c in range(3):
        arr2Value[r][c]=imgColor(arr2[r][c])

# pink to wight -> 0     RED
# wight to pink -> 1    BLUE
# black to color -> 2   GREEN
# color to black -> 3   YELLOW
# no change -> 4

result=[[0]*3]*3

for r in range(3):
    for c in range(3):
        if (arr1Value[r][c] == 0):

            if(arr2Value[r][c]==0):
                result[r][c] = 4
            if(arr2Value[r][c]==1):
                result[r][c]=2
            if(arr2Value[r][c]==2):
                result[r][c] = 2


        if (arr1Value[r][c] == 1):
            if (arr2Value[r][c] == 0):
                result[r][c] = 3
            if (arr2Value[r][c] == 1):
                result[r][c] = 4
            if (arr2Value[r][c] == 2):
                result[r][c] = 0


        if (arr1Value[r][c] == 2):
            if (arr2Value[r][c] == 0):
                result[r][c] = 3
            if (arr2Value[r][c] == 1):
                result[r][c] = 1
            if (arr2Value[r][c] == 2):
                result[r][c] = 4


#reseting img2 value
img2=cv2.imread("y.jpg")
img2=cv2.resize(img2,(600,600))

for r in range(3):

    for c in range(3):

        color=(0,0,0)
        if (result[r][c]==0):
            color=0,100,100
            img2 = cv2.rectangle(img2, (int(c * 200), int(r * 200)), (int(c * 200 + 200), int(r * 200 + 200)), color,10)


        if (result[r][c]==1):
            color=240,100,100
            img2 = cv2.rectangle(img2, (int(c * 200), int(r * 200)), (int(c * 200 + 200), int(r * 200 + 200)), color, 10)


        if (result[r][c]==2):
            color=100,100,100
            img2 = cv2.rectangle(img2, (int(c * 200), int(r * 200)), (int(c * 200 + 200), int(r * 200 + 200)), color, 10)

        if (result[r][c]==3):
            color=60,95,100
            img2 = cv2.rectangle(img2, (int(c * 200), int(r * 200)), (int(c * 200 + 200), int(r * 200 + 200)), color,10)





cv2.imshow("result",img2)

cv2.waitKey()



