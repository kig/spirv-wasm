const fs = require('fs');

function getMatchingDef(name, defaultValue) {
    const re = new RegExp(`(^|;|\\s)${name}\\s*=\\s*(\\d+)\\s*;`);
    const m = source.match(re);
    if (m) {
        source = source.replace(re, '');
        return parseInt(m[2]);
    }
    return defaultValue;
}

var source = fs.readFileSync(process.argv[2]).toString();

const threadLocalCount = getMatchingDef("ThreadLocalCount", 64);
const threadGroupCount = getMatchingDef("ThreadGroupCount", 256);
const heapSize = getMatchingDef("HeapSize", 4096);
const toIOSize = getMatchingDef("ToIOSize", 4096);
const fromIOSize = getMatchingDef("FromIOSize", 4096);

const totalHeapSize = getMatchingDef("TotalHeapSize", heapSize * threadLocalCount * threadGroupCount);
const totalToIOSize = getMatchingDef("TotalToIOSize", toIOSize * threadLocalCount * threadGroupCount);
const totalFromIOSize = getMatchingDef("TotalFromIOSize", fromIOSize * threadLocalCount * threadGroupCount);

const segments = source.replace(/^# .*/mg, '').split(/("|')/g);

let inString = false;
let inChar = false;
let lastSegment = '';
let stringSegments = [];

const output = [];

const globals = [];
const globalsOut = [];
let globalBytes = 0;

// FIXME write the byte data to the heap's global section
// - [x] Add the string data to the SPIR-V file as a source section
// - [x] Parse out the string at shader load time, copy to start of heap buffer
// - [x] Move globals section from end of heap to start of heap
// - [x] Add command line arguments after the globals

// TODO Shader controls heap size
// - [x] Parse heap size and workgroup size info from shader add to SPIR-V as a source section
// - [x] Parse heap size and workgroup size at shader load time to configure runner app
// - [ ] Add optparse to runner for overriding shader params and setting verbose and timings flags

// - [x] Add #!/usr/bin/gls to top of shader files to run as script
// - [ ] Include path /usr/lib/script-v/
// - [ ] Shader cache in ~/.script-v/

for (segment of segments) {
	if (segment === '"' && lastSegment[lastSegment.length-1] !== '\\') {
		inString = !inString;
		if (!inString) {
			const str = stringSegments.join('');
			const buf = Buffer.from(JSON.parse('"'+str+'"'));
			const len = buf.length;
			const v = `_global_${globals.length}_`;
			output.push(v);
			globals.push(`ivec2 ${v} = HeapGlobalsOffset + ivec2(${globalBytes}, ${globalBytes+len});`);
			globalBytes += len;
			globalsOut.push(buf);
		}
		stringSegments = [];
	} else if (inString) {
		stringSegments.push(segment);
	} else if (segment === "'" && lastSegment[lastSegment.length-1] !== '\\') {
		inChar = !inChar;
		if (!inChar) {
			const str = eval("'" + stringSegments.join('') + "'");
			if (str.length == 4) {
				output.push(`${Buffer.from(str).readInt32LE(0)}`);
			} else if (str.length == 1) {
				output.push(`toChar(${Buffer.from(str)[0]})`);
			}
		}
		stringSegments = [];
	} else if (inChar) {
		stringSegments.push(segment);
	} else {
		output.push(segment);
	}
	lastSegment = segment;
}

let outputString = `#version 450
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : require

int32_t HeapSize = ${heapSize};
int32_t FromIOSize = ${fromIOSize};
int32_t ToIOSize = ${toIOSize};

int32_t TotalHeapSize = ${totalHeapSize};
int32_t TotalFromIOSize = ${totalFromIOSize};
int32_t TotalToIOSize = ${totalToIOSize};

int32_t HeapGlobalsOffset = ${totalHeapSize + 8};

layout ( local_size_x = ${threadLocalCount}, local_size_y = 1, local_size_z = 1 ) in;

` + output.join('').replace('%%GLOBALS%%', globals.join('\n'));

// - If shader has no void main() {}, wrap the tail of the file from }(.*)EOF in }\nvoid main() { $1 }
// FIXME Do this on AST level... Find the last definition (function, struct, etc.) instead of last }
if (!/void\s+main\s*\(\s*\)/m.test(outputString)) {
    const idx = outputString.lastIndexOf('}') + 1;
    outputString = outputString.slice(0,idx) + '\nvoid main() {\n' + outputString.slice(idx) + '\n}\n';
}

// Create SPV globals suffix

function OpSourceExtUInt32(tag, num) {
    const tagbuf = Buffer.from(tag);
    const outbuf = Buffer.alloc(12);
    outbuf.writeUInt16LE(4, 0);
    outbuf.writeUInt16LE((outbuf.byteLength/4), 2);
    outbuf.set(tagbuf, 4);
    outbuf.writeUInt32LE(num, 8);
    return outbuf;
}

function OpSourceExtStr(tag, str) {
    const buf = Buffer.from(str);
    const tagbuf = Buffer.from(tag);
    const outbuf = Buffer.alloc(8+Math.ceil(buf.length/4)*4);
    outbuf.writeUInt16LE(4, 0);
    outbuf.writeUInt16LE((outbuf.byteLength/4), 2);
    outbuf.set(tagbuf, 4);
    outbuf.set(buf, 8);
    return outbuf;
}

{
    const output = [];
    output.push(OpSourceExtStr("glo=", Buffer.concat(globalsOut)));

    output.push(OpSourceExtUInt32("tgc=", threadGroupCount));

    output.push(OpSourceExtUInt32("hsz=", heapSize));
    output.push(OpSourceExtUInt32("ths=", totalHeapSize));

    output.push(OpSourceExtUInt32("fis=", fromIOSize));
    output.push(OpSourceExtUInt32("tfi=", totalFromIOSize));

    output.push(OpSourceExtUInt32("tis=", toIOSize));
    output.push(OpSourceExtUInt32("tti=", totalToIOSize));

    fs.writeFileSync(process.argv[2] + '.defs.spv', Buffer.concat(output));
}

// Write out the files
fs.writeFileSync(process.argv[2] + '.comp', outputString);
