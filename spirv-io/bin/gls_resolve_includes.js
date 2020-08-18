const fs = require('fs');
const cp = require('child_process');
const crypto = require('crypto');

// Resolve and inline all #include foo things

function resolveIncludes(source) {
    const resolvedSrc = source.replace(/^\s*#include\s+<(https:.*)>\s*@?\s*([a-zA-Z0-9]+)?$/mg, (match, url, hash) => {
        /*
        if (url.startsWith('"')) { // Local file
        } else if (url.startsWith('<https://')){ // URL
        } else { // System library
        }
        */
        if (hash) hash = hash.toLowerCase();
        const localPath = `${process.env.HOME}/.gls/cache/lib/${encodeURIComponent(url)}${hash ? `%${encodeURIComponent(hash)}` : '%0000'}`;
        if (!fs.existsSync(localPath)) {
            cp.execFileSync('mkdir', ['-p', `${process.env.HOME}/.gls/cache/lib`]);
            fs.writeFileSync(localPath, "");
            const urlSource = cp.execFileSync('curl', ['--silent', url]).toString();
            const contentHash = crypto.createHash('sha256').update(urlSource).digest('hex');
            if (hash && contentHash !== hash) {
                console.error(`Downloaded ${url}\nERROR: Library file has been tampered with!\n       Downloaded content hash ${contentHash} differs from requested ${hash}`);
                process.exit(1);
            } else {
                console.error(`Downloaded ${url} - content hash ${contentHash}`);
            }
            const resolved = resolveIncludes(urlSource);
            const hashLocalPath = `${process.env.HOME}/.gls/cache/lib/${encodeURIComponent(url)}%${encodeURIComponent(contentHash)}`;
            fs.writeFileSync(hashLocalPath, resolved);
            fs.writeFileSync(localPath, resolved);
        }
        return `#include "${localPath}"`;
    });
    return resolvedSrc;
}

const srcBuf = Buffer.alloc(1048576);
const srcLen = fs.readSync(process.stdin.fd, srcBuf);
const source = srcBuf.slice(0, srcLen).toString();

const resolvedSrc = resolveIncludes(source);

fs.writeSync(process.stdout.fd, resolvedSrc);
