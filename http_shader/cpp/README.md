C++ version of the key-value store shader. Compiled to C++ with spirv-cross.

The atomics implementation seems to not be working too well.

Build and run:

```bash
(cd .. && sh build_cpp.sh)
sh run.sh
# 200 OK HTTP/1.1
# content-type: text/plain
# 
# OK.
# 200 OK HTTP/1.1
# content-type:  text/html
# 
# <html><body>This is document number 1.</body></html>
# 200 OK HTTP/1.1
# content-type: text/plain
# 
# BLK
# 200 OK HTTP/1.1
# content-type:  text/html
# 
# <html><body>This is document number 3.</body></html>
# 200 OK HTTP/1.1
# content-type: text/plain
# 
# OK.
# 200 OK HTTP/1.1
# content-type:  text/html
# 
# <html><body>This is document number 5.</body></html>
# 200 OK HTTP/1.1
# content-type: text/plain
# 
# OK.
# 200 OK HTTP/1.1
# content-type:  text/html
# 
# <html><body>This is 2067 spam-post 2067 number 2067.</body></html>
# 200 OK HTTP/1.1
# content-type: text/plain
# 
# BLK
# 200 OK HTTP/1.1
# content-type:  text/html
# 
# <html><body>This is document number 9.</body></html>
# 
# Elapsed: 3180 ms
# Million requests per second: 16.487
# 
# 40.38user 0.91system 0:03.65elapsed 1129%CPU (0avgtext+0avgdata 1576176maxresident)k
# 0inputs+0outputs (0major+393371minor)pagefaults 0swaps
```
