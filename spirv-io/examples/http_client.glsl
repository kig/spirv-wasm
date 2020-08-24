#include <file.glsl>
#include <hashtable.glsl>

ThreadLocalCount = 1;
ThreadGroupCount = 1;

HeapSize = 16777216;
FromIOSize = 16777216;
ToIOSize = 16777216;

void main() {
    if (ThreadId == 0) {
        string url = "https://github.com/plotly/datasets/raw/master/tips.csv";
        log(concat("Downloading ", url));
        awaitIO(runCmd(concat("curl -s -L -O ", url)));
        log("Downloaded");
        uint64_t sz = statSync("tips.csv").st_size;
        log(concat("File size: ", str(sz)));
        string csv = readSync("tips.csv", malloc(sz));
        stringArray lines = split(csv, '\n');
        i32map counts = i32hAlloc(16);
        i32map means = f32hAlloc(16);
        for (int i = 1; i < arrLen(lines); i++) {
            int32_t count;
            float mean;
            int32_t size = -1;
            FREE(
                stringArray fields = split(aGet(lines, i), ',');
                if (arrLen(fields) == 7) {
                    float total_bill = parsef32(aGet(fields, 0));
                    float tip = parsef32(aGet(fields, 1));
                    size = parsei32(aGet(fields, 6));
                    float tip_pct = 100.0 * tip / total_bill;
                    if (!i32hGet(counts, size, count)) {
                        count = 0;
                        mean = 0.0;
                    } else {
                        f32hGet(means, size, mean);
                    }
                    count += 1;
                    mean = (mean * float(count-1) + tip_pct) / float(count);
                }
            );
            if (size != -1) {
                i32hSet(counts, size, count);
                f32hSet(means, size, mean);
            }
        }
        i32array sizes = f32hKeys(means);
        i32sort(sizes);
        for (int i = 0; i < i32len(sizes); i++) {
            int32_t size = i32get(sizes, i);
            float mean;
            if (f32hGet(means, size, mean)) {
                FREE_ALL( log(concat("size: ", str(size), " tip_pct: ", str(mean))) );
            }
        }
    }
}

