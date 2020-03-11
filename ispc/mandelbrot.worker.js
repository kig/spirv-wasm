let instance = null;
let heap = null;

WebAssembly.instantiateStreaming(fetch('mandelbrot.wasm'))
    .then(obj => {
	    obj.instance.exports.memory.grow(600);
	    heap = new Int32Array(obj.instance.exports.memory.buffer);
	    instance = obj.instance;
	    postMessage("ready");
	});

onmessage = function(ev) {
    const {x0, y0, x1, y1, width, height, maxIterations, buf} = ev.data;
    instance.exports.mandelbrot_ispc(x0, y0, x1, y1, width, height, maxIterations, 0);
    const u8 = new Uint8Array(buf);
    for (let i = 0; i < width*height; i++) {
	u8[i*4 + 0] = heap[i] * 5;
	u8[i*4 + 1] = heap[i] * 5;
        u8[i*4 + 2] = heap[i] * 5;
	u8[i*4 + 3] = 255;
    }
    postMessage(buf, [buf]);
}
