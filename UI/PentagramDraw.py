#Pentagram Challenge - www.101computing.net/pentagram-challenge/
import turtle, math
import matplotlib.pyplot as plt
import sys, time, os, datetime
from platform import python_version
myPen = turtle.Turtle()
myPen.shape("arrow")
myPen.pencolor("purple")
myPen.pensize(2)
myPen.speed(1000)

#A Procedure to draw a polygon from a list of vertices.
def drawPolygon(polygon):
 myPen.penup()
 myPen.goto(polygon[0][0],polygon[0][1])
 myPen.pendown()
 
 for i in range(1,len(polygon)):
    myPen.goto(polygon[i][0],polygon[i][1])
 
 myPen.goto(polygon[0][0],polygon[0][1])
 
Dots1Inch_height=96
Dots1Inch_width=96

#A polygon can be stored as a list of vertices 
pentagon=[]
R = 25.85

plt.title("Pentagon"+time.strftime("%Y%m%d_%H%M%S"))
for n in range(0,5):
  x = R*math.cos(math.radians(90+n*72))
  y = R*math.sin(math.radians(90+n*72))
  pentagon.append([x,y])
  plt.plot(pentagon)

plt.savefig(time.strftime("%Y%m%d_%H%M%S")+"_Pentagon.png",format="png",dpi=Dots1Inch_height)
plt.cla()

drawPolygon(pentagon)
myPen.hideturtle()

robot_vertices = [[(R*(math.sin(math.radians(90+0*72)))),                       (R*(math.cos(math.radians(90+0*72))))], 
                    [(R*(math.sin(math.radians(90+1*72)))),    (R*(math.cos(math.radians(90+1*72))))], 
                    [(R*(math.sin(math.radians(90+2*72)))), (R*(math.cos(math.radians(90+2*72))))],
                    [(R*(math.sin(math.radians(90+3*72)))),  (R*(math.cos(math.radians(90+3*72))))], 
                    [(R*(math.sin(math.radians(90+4*72)))),     (R*(math.cos(math.radians(90+4*72))))]]
plt.title("Pentagon"+time.strftime("%Y%m%d_%H%M%S"))
plt.plot(robot_vertices)
plt.pause(2) # Sample time in seconds (s)
drawPolygon(robot_vertices)
myPen.hideturtle()