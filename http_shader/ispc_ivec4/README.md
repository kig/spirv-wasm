# A small in-memory key-value store

Send a HTTP `GET /xxxxxxx` to get the value at key `xxxxxxx`. 

Send a `POST /xxxxxxx` to set the value at key `xxxxxxx` to the POST body. The start of the body should be the content-type, followed by `\r\n\r\n`. 
You can add other headers after the content-type if you want. See `httpd_ispc_ivec4.cpp` for an example.

Access to the values is protected by a per-object atomic "mutex". Only one write request can execute at a time, other simultaneous writes are rejected.
If you try to write while you have readers, the write will fail. If you try to read when a write is in process, the read will fail.

## Build & run

```bash
(cd .. && sh build_ispc_ivec4.sh)
sh run.sh
# 200 OK HTTP/1.1
# content-type: text/plain
# 
# BLK
# 200 OK HTTP/1.1
# content-type:  text/html
# 
# <html><body>This is 150281 spam-post 150281 number 150281.</body></html>
# 200 OK HTTP/1.1
# content-type: text/plain
# 
# BLK
# 200 OK HTTP/1.1
# content-type:  text/html
# 
# <html><body>This is 248863 spam-post 248863 number 248863.</body></html>
# 200 OK HTTP/1.1
# content-type: text/plain
# 
# OK.
# 200 OK HTTP/1.1
# content-type:  text/html
# 
# <html><body>This is 18435 spam-post 18435 number 18435.</body></html>
# 200 OK HTTP/1.1
# content-type: text/plain
# 
# BLK
# 200 OK HTTP/1.1
# content-type:  text/html
# 
# <html><body>This is 81927 spam-post 81927 number 81927.</body></html>
# 200 OK HTTP/1.1
# content-type: text/plain
# 
# BLK
# 200 OK HTTP/1.1
# content-type:  text/html
# 
# <html><body>This is 3079 spam-post 3079 number 3079.</body></html>
# 
# Elapsed: 19885 ms
# Million requests per second: 26.366
# 
# 606.42user 0.93system 0:20.34elapsed 2986%CPU (0avgtext+0avgdata 1576412maxresident)k
# 0inputs+0outputs (0major+393407minor)pagefaults 0swaps
```
