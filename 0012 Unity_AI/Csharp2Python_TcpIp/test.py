import struct

name = "kim"
subject = "math"
grade = "33"
grade = int(grade)
memo = "goodluck"

name = bytes(name,'utf-8')
subject = bytes(subject,'utf-8')
memo = bytes(memo,'utf-8')
senddata = struct.pack('20s20si100s',name,subject,grade,memo)
print(senddata)