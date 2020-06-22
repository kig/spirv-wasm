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

for (segment of segments) {
	if (segment === '"' && lastSegment[lastSegment.length-1] !== '\\') {
		inString = !inString;
		if (!inString) {
			const str = stringSegments.join('');
			const buf = Buffer.from(JSON.parse('"'+str+'"'));
			const len = buf.length;
			const v = `_global_${globals.length}_`;
			const assigns = [];
			for (let i = 0; i < len; i++) assigns.push(`setC(${v}, ${i}, int8_t(${buf[i]}))`);
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
				output.push(`int8_t(${Buffer.from(str)[0]})`);
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

console.log(output.join('').replace('%%GLOBALS%%', globals.join('\n')).replace('%%INIT%%', init.join('\n')));
