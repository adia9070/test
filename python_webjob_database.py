import sys  
sys.path.append(r"D:\home\site\wwwroot\env\Lib\site-packages")
print(sys.version)
try:
	import pyodbc
except:
	print(sys.version)
