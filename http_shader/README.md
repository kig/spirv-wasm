# http_shader

GLSL shaders that parse HTTP requests and write out HTTP responses. 

There are three different ways here:

* `httpd.glsl` - Turn a inputBuffer of 8-bit ASCII characters into a buffer of ints, parse requests in the int buffer, create responses in another buffer of ints, convert the response buffer into 8-bit char outputBuffer.
* `httpd_int.glsl` - Same thing but when converting the 8-bit chars to ints, turn them into SOA format (`[req_0_0, req_1_0, req_2_0, req_3_0, ... req_31_0, req_0_1, ...]`) for some faster-going
* `httpd_ivec4.glsl` - Just deal with the inputBuffer & outputBuffer directly. Using ivec4s because doesn't that sound painful? The ISPC version does >100 million "Hello, world!" requests per second on a TR2950X 16-core.

I ... found out that SPIR-V in Vulkan 1.2 supports 8-bit ints as a native type. I want to use those instead of writing helper functions to get/set individual bytes in ivec4s.

In a shocking turn of events, running these on the CPU via SPIR-V to ISPC performs better than the GPU. Even after removing the buffer uploads and downloads.
