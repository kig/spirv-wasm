# A small in-memory key-value store

Send a HTTP `GET /xxxxxxx` to get the value at key `xxxxxxx`. 

Send a `POST /xxxxxxx` to set the value at key `xxxxxxx` to the POST body. The start of the body should be the content-type, followed by `\r\n\r\n`. 
You can add other headers after the content-type if you want. See `vulkanRunner.cpp` for an example.

Access to the values is protected by a per-object atomic "mutex". Only one write request can execute at a time, other simultaneous writes are rejected.
If you try to write while you have readers, the write will fail. If you try to read when a write is in process, the read will fail.

## Build & run

```bash
(cd .. && sh build_vulkan.sh)
sh run.sh
# 200 OK HTTP/1.1
# content-type: text/plain
# 
# BLK
# 200 OK HTTP/1.1
# content-type:  text/html
# 
# <html><body>This is 20481 spam-post 20481 number 20481.</body></html>
# 200 OK HTTP/1.1
# content-type: text/plain
# 
# BLK
# 200 OK HTTP/1.1
# content-type:  text/html
# 
# <html><body>This is 97283 spam-post 97283 number 97283.</body></html>
# 200 OK HTTP/1.1
# content-type: text/plain
# 
# BLK
# 200 OK HTTP/1.1
# content-type:  text/html
# 
# <html><body>This is 286725 spam-post 286725 number 286725.</body></html>
# 200 OK HTTP/1.1
# content-type: text/plain
# 
# BLK
# 200 OK HTTP/1.1
# content-type:  text/html
# 
# <html><body>This is 138247 spam-post 138247 number 138247.</body></html>
# 200 OK HTTP/1.1
# content-type: text/plain
# 
# BLK
# 200 OK HTTP/1.1
# content-type:  text/html
# 
# <html><body>This is 496649 spam-post 496649 number 496649.</body></html>
# 
# Elapsed: 6731 ms
# Million requests per second: 7.789
# 
# 0.27user 1.03system 0:08.06elapsed 16%CPU (0avgtext+0avgdata 3217080maxresident)k
# 0inputs+40outputs (2major+267993minor)pagefaults 0swaps
```
