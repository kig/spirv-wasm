const fs = require('fs');

const source = fs.readFileSync(0);

const segments = source.toString().replace(/^# .*/mg, '').split(/("|')/g);

let inString = false;
let inChar = false;
let lastSegment = '';
let stringSegments = [];

const output = [];

for (segment of segments) {
	if (segment === '"' && lastSegment[lastSegment.length-1] !== '\\') {
		inString = !inString;
		if (!inString) {
			const str = stringSegments.join('');
			output.push(`{${Buffer.from(JSON.parse('"'+str+'"')).join(",")}}`);
		}
		stringSegments = [];
	} else if (inString) {
		stringSegments.push(segment);
	} else if (segment === "'" && lastSegment[lastSegment.length-1] !== '\\') {
		inChar = !inChar;
		if (!inChar) {
			const str = stringSegments.join('');
			output.push(`${Buffer.from(eval("'"+str+"'")).readInt32LE(0)}`);
		}
		stringSegments = [];
	} else if (inChar) {
		stringSegments.push(segment);
	} else {
		output.push(segment);
	}
	lastSegment = segment;
}

console.log(output.join(''));
