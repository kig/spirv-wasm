const fs = require('fs');

const source = fs.readFileSync(process.argv[2]).toString();

const testFuncs = {};
let i = 0;

const segs = source.split(/(\/\*T)|(\*\/)/y);

let inTest = false;
let testBody = '';
for (let i = 0; i < segs.length; i++) {
    const seg = segs[i];
    if (seg === '\/*T') {
        inTest = true;
        testBody = '';
    } else if (seg === '*\/' && inTest) {
        inTest = false;
        const nextSeg = segs[i+1];
        if (nextSeg) {
            const m = nextSeg.match(/^\s*\S+\s+([^(]+)/m);
            if (m) {
                let name = 'test_'+m[1].trim();
                while (testFuncs[name]) {
                    name += '_';
                }
                testBody = testBody.replace(
                    /^\s*(\S+)\s*(<=?|>=?|==|!=)\s*([^;]+);\s*$/mg,
                    (m, lv, cmp, rv) =>
                        `    assert(${lv} ${cmp} ${rv});`
                );
                testFuncs[name] = `void ${name}() {\n${testBody}\n}`;
            }
        }
    } else if (inTest && seg) {
        testBody = seg;
    }
}

let testSource = [`
#include <assert.glsl>
#include "${process.argv[2]}"

ThreadLocalCount = 1;
ThreadGroupCount = 1;

HeapSize = 16777216;
ToIOSize = 16777216;
FromIOSize = 16777216;
`];
let testMain = `void main() {
`;
for (const funcName in testFuncs) {
    testMain += `    FREE_ALL(${funcName}());\n`
    testSource.push(testFuncs[funcName]);
}
testMain += '}';
testSource.push(testMain);

console.log(testSource.join("\n\n"));
