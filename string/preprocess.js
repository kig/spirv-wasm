const fs = require('fs');

const source = fs.readFileSync(0);

const segments = source.toString().replace(/^# .*/mg, '').split(/("|')/g);

let inString = false;
let inChar = false;
let lastSegment = '';
let stringSegments = [];

const output = [];

const globals = [];
const init = [];

// FIXME write the byte data to the heap's global section
// - Add the string data to the SPIR-V file as a source section
// - Parse out the string at shader load time, copy to start of heap buffer
// - Move globals section from end of heap to start of heap
// - Add command line arguments after the globals

// TODO Shader controls heap size
// - Parse heap size and workgroup size info from shader add to SPIR-V as a source section
// - Parse heap size and workgroup size at shader load time to configure runner app
// - Add optparse to runner for overriding shader params and setting verbose and timings flags

// - Add #!/usr/bin/gls to top of shader files to run as script
// - Include path /usr/lib/script-v/
// - Shader cache in ~/.script-v/

for (segment of segments) {
	if (segment === '"' && lastSegment[lastSegment.length-1] !== '\\') {
		inString = !inString;
		if (!inString) {
			const str = stringSegments.join('');
			const buf = Buffer.from(JSON.parse('"'+str+'"'));
			const len = buf.length;
			const v = `_global_${globals.length}_`;
			const assigns = [];
			for (let i = 0; i < len; i++) assigns.push(`setC(${v}, ${i}, toChar(${buf[i]}))`);
			output.push(v);
			globals.push(`ivec2 ${v} = malloc(${len});`);
			init.push(`${assigns.join(';')};`);
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

console.log(`#version 450
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : require

`);
let outputString = output.join('').replace('%%GLOBALS%%', globals.join('\n')).replace('%%INIT%%', init.join('\n'));
// - If shader has no void main() {}, wrap the tail of the file from }(.*)EOF in }\nvoid main() { $1 }
if (!/void\s+main\s*\(\s*\)/m.test(outputString)) {
    const idx = outputString.lastIndexOf('}') + 1;
    outputString = outputString.slice(0,idx) + '\nvoid main() {\n' + outputString.slice(idx) + '\n}\n';
}
// Add initGlobals call if missing one. Should be deprecated after the above FIXME
if (!/initGlobals\(\);/.test(outputString)) {
    outputString = outputString.replace(/void main\(\) {/, 'void main() {initGlobals();');
}
console.log(outputString);
