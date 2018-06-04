import os 
import psutil  
process = psutil.Process(os.getpid())   
import resource   

def mem(string=''): 
    rss = process.memory_info().rss
    print(rss/float(2**20),'MB',rss/float(2**30),'GB',string) 
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(rss/float(2**10),'MB',rss/float(2**20),'GB',string) 
 
