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

const groupHeapSize = getMatchingDef("GroupHeapSize", heapSize * threadLocalCount);
const groupToIOSize = getMatchingDef("GroupToIOSize", toIOSize * threadLocalCount);
const groupFromIOSize = getMatchingDef("GroupFromIOSize", fromIOSize * threadLocalCount);

const totalHeapSize = getMatchingDef("TotalHeapSize", groupHeapSize * threadGroupCount);
const totalToIOSize = getMatchingDef("TotalToIOSize", groupToIOSize * threadGroupCount);
const totalFromIOSize = getMatchingDef("TotalFromIOSize", groupFromIOSize * threadGroupCount);

const segments = source.replace(/^# .*/mg, '').split(/("|'|[fui]\d\d?{|})/g);

let inString = false;
let inChar = false;
let inArray = false;
let arrayType = '';
let lastSegment = '';
let stringSegments = [];
let arraySegments = [];

const output = [];

const globals = [];
const arrayGlobals = [];
const globalsOut = [];
let globalBytes = 0;

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
	} else if (/^\S+{$/.test(segment)) {
	    inArray = true;
	    arrayType = segment.slice(0,-1);
    } else if (inArray && segment === '}') {
        inArray = false;
		const str = arraySegments.join('');
		const v = `_array_global_${arrayGlobals.length}_`;
		output.push(v);
		let bitSize = parseInt(arrayType.match(/64|32|16|8/)[0]);
		let elementCount = 1;
		let byteSize = elementCount * bitSize / 8;
		// Align on byteSize
		const newGlobalBytes = Math.ceil(globalBytes / byteSize) * byteSize;
		globalsOut.push(Buffer.alloc(newGlobalBytes - globalBytes)); // Add padding to global buffer.
		globalBytes = newGlobalBytes;
		const sz = byteSize;
		const arrayElems = str.split(",").filter(s => !/^\s*$/.test(s));
		const buf = Buffer.alloc(arrayElems.length * byteSize);
		arrayGlobals.push(`${arrayType}array ${v} = ${arrayType}array((HeapGlobalsOffset + ${globalBytes})/${sz}, (HeapGlobalsOffset + ${globalBytes+buf.byteLength})/${sz});`);
		let i = 0;
		arrayElems.forEach((el) => {
    		switch(arrayType) {
    		    case 'f32': buf.writeFloatLE(parseFloat(el),i); i+=4; break;
    		    case 'f64': buf.writeDoubleLE(parseFloat(el),i); i+=8; break;
    		    case 'i8': buf.writeInt8LE(iparseInt(el),i); i+=1; break;
    		    case 'i16': buf.writeInt16LE(parseInt(el),i); i+=2; break;
    		    case 'i32': buf.writeInt32LE(parseInt(el),i); i+=4; break;
    		    case 'i64': buf.writeBigInt64LE(BigInt(el),i); i+=8; break;
    		    case 'u8': buf.writeUInt8LE(parseInt(el),i); i+=1; break;
    		    case 'u16': buf.writeUInt16LE(parseInt(el),i); i+=2; break;
    		    case 'u32': buf.writeUInt32LE(parseInt(el),i); i+=4; break;
    		    case 'u64': buf.writeBigUInt64LE(BigInt(el),i); i+=8; break;
    		}
		});
 		globalBytes += buf.byteLength;
		globalsOut.push(buf);
		arraySegments = [];
	} else if (inArray) {
		arraySegments.push(segment);
	} else {
		output.push(segment);
	}
	lastSegment = segment;
}

let outputString = `#version 450
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_ARB_shader_clock : require
#extension GL_EXT_shader_realtime_clock : require

int32_t HeapSize = ${heapSize};
int32_t FromIOSize = ${fromIOSize};
int32_t ToIOSize = ${toIOSize};

int32_t GroupHeapSize = ${groupHeapSize};
int32_t GroupFromIOSize = ${groupFromIOSize};
int32_t GroupToIOSize = ${groupToIOSize};

int32_t TotalHeapSize = ${totalHeapSize};
int32_t TotalFromIOSize = ${totalFromIOSize};
int32_t TotalToIOSize = ${totalToIOSize};

int32_t HeapGlobalsOffset = ${totalHeapSize + 8};

layout ( local_size_x = ${threadLocalCount}, local_size_y = 1, local_size_z = 1 ) in;

` + output.join('').replace('%%GLOBALS%%', globals.join('\n')).replace('%%ARRAYGLOBALS%%', arrayGlobals.join('\n'));

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
