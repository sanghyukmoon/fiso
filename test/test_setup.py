import numpy as n
import time
time.t0 = time.time()
def timer(string):
    time.t1 = time.time()
    print(time.t1-time.t0,string)
    time.t0 = time.t1
size = 256
shape = [size]*3
x,y,z = n.indices(shape)
# set up potential with 4 point sources

def point(x0,y0,z0):
    radius = n.sqrt((x-x0)**2 + (y-y0)**2 + (z-z0)**2) + 1.
    return -1./radius

f1 = 0.25*size
f2 = 0.75*size
df = 0.05*size
zlist = [0.5*size]*4
potential = point(f1,f1,zlist[0]) +\
point(f1,f2,zlist[1]) +\
point(f2,f1,zlist[2]) +\
0.25*point(f2,f2+df,zlist[3]) +\
0.25*point(f2+df,f2,zlist[3]) +\
0.25*point(f2,f2,zlist[3]) +\
0.25*point(f2+df,f2+df,zlist[3]) 

potential[int(f1)+5,int(f1),int(zlist[0])] *= 2.0
potential[int(f1)-5,int(f1),int(zlist[0])] *= 2.0
