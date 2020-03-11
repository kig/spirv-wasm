function makeWorker(onready) {
    const worker = new Worker('mandelbrot.worker.js');
    const queue = [];
    worker.onmessage = (ev) => {
	if (ev.data === 'ready') onready();
	else {
	    queue.shift()(ev.data);
	}
    };
    worker.call = (args, transfers) => {
	return new Promise((resolve, reject) => {
		queue.push(resolve);
		worker.postMessage(args, transfers);
	    });
    };
    return worker;
}

let readyCount = 0;

const onready = () => {
    readyCount++;
    if (workerCount === readyCount) init();
}

const workerCount = navigator.hardwareConcurrency;
const workers = new Array(workerCount).fill().map(x => makeWorker(onready));


async function init() {
    let width, segHeight, height, bufs, id, resized;

    const canvas = document.createElement('canvas');
    document.body.append(canvas);
    const ctx = canvas.getContext('2d');

    window.onresize = () => {
	resized = true;
	const width = window.innerWidth;
	const proposedHeight = innerHeight;
    
	segHeight = 40;
        segCount = Math.ceil(proposedHeight / segHeight);
	const height = segCount * segHeight;
	bufs = new Array(segCount).fill().map(x => new ArrayBuffer(width*segHeight*4));
	canvas.width = width;
	canvas.height = height;
	id = ctx.createImageData(width, height);
    };

    window.onresize();

    const X0 = -2.5, X1 = 1, Y0 = -1, Y1 = 1;
    const maxIterations = 50;
    
    const segments = [];

    const tick = async () => {
	resized = false;
	const aspect = canvas.width / canvas.height * 0.5;
	const t = Date.now() / 2000;
	const z = (1.01 + Math.cos(t * 2.1234));
	const x0 = (X0*aspect + Math.sin(t)/z/1.5) * z
	const x1 = (X1*aspect + Math.sin(t)/z/1.5) * z
	const y0 = (Y0 + Math.cos(t*Math.sqrt(2))/z/1.5) * z
	const y1 = (Y1 + Math.cos(t*Math.sqrt(2))/z/1.5) * z
	for (let y = 0, i = 0, w = 0; y < canvas.height; y += segHeight, i++, w++) {
	    if (resized) break;
	    const f0 = y/canvas.height;
	    const f1 = (y+segHeight)/canvas.height;
	    const ry0 = y0 * (1-f0) + y1 * f0;
	    const ry1 = y0 * (1-f1) + y1 * f1;
	    if (w === workers.length) w = 0;
	    segments[i] = workers[w].call({x0, y0: ry0, x1, y1: ry1, width: canvas.width, height: segHeight, maxIterations, buf: bufs[i]}, [bufs[i]]);
	}
	for (let i = 0; i < segments.length; i++) {
	    const buf = await segments[i];
	    if (resized) break;
	    id.data.set(new Uint8Array(buf), i * segHeight * canvas.width * 4);
	    bufs[i] = buf;
	}
	if (!resized) ctx.putImageData(id, 0, 0);
	requestAnimationFrame(tick);
    };

    tick();
}