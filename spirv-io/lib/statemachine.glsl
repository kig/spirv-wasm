struct stateMachine {
    ptr_t statePtr;
    stringArray attrs;
    ptr_t heapPtrPtr;
    ptr_t fromIOPtrPtr;
    ptr_t toIOPtrPtr;
};

stateMachine loadStateMachine(int initialState) {
    stateMachine m = stateMachine(
        heapPtr/4 + 1,
        stringArray(heapPtr/4 + 2, heapPtr/4 + 28),
        heapPtr/4 + 29,
        heapPtr/4 + 30,
        heapPtr/4 + 31
    );
    if (i32heap[heapPtr/4] != 0x57A7E0FC) {
        i32heap[heapPtr/4] = 0x57A7E0FC;
        i32heap[m.statePtr] = initialState;
        for (ptr_t i = m.attrs.x; i < m.attrs.y; i++) {
            i32heap[i] = 0;
        }
        i32heap[m.heapPtrPtr] = heapPtr + 32*4;
        i32heap[m.fromIOPtrPtr] = fromIOPtr;
        i32heap[m.toIOPtrPtr] = toIOPtr;
    }
    heapPtr = i32heap[m.heapPtrPtr];
    fromIOPtr = i32heap[m.fromIOPtrPtr];
    toIOPtr = i32heap[m.toIOPtrPtr];
    return m;
}

void saveStateMachine(stateMachine m) {
    i32heap[m.heapPtrPtr] = heapPtr;
    i32heap[m.toIOPtrPtr] = toIOPtr;
    i32heap[m.fromIOPtrPtr] = fromIOPtr;
}

int getState(stateMachine m) {
    return i32heap[m.statePtr];
}

void setState(stateMachine m, int state) {
    i32heap[m.statePtr] = state;
}

void setAttr(stateMachine m, int key, string value) {
    aSet(m.attrs, key, value);
}

string getAttr(stateMachine m, int key) {
    return aGet(m.attrs, key);
}

void setAttr(stateMachine m, int key, io value) {
    aSet(m.attrs, key, string(value.index, value.heapBufStart));
}

void setAttr(stateMachine m, int key, int32_t value) {
    aSet(m.attrs, key, string(value, 0));
}

io getIOAttr(stateMachine m, int key) {
    string s = aGet(m.attrs, key);
    return io(s.x, s.y);
}

int32_t getI32Attr(stateMachine m, int key) {
    string s = aGet(m.attrs, key);
    return int32_t(s.x);
}
