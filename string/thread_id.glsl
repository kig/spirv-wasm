int ThreadGroupCount = int(gl_NumWorkGroups.x);
int ThreadLocalCount = int(gl_WorkGroupSize.x);
int ThreadCount = ThreadGroupCount * ThreadLocalCount;
int ThreadID = int(gl_GlobalInvocationID.x);
int ThreadGroupID = int(gl_WorkGroupID.x);
int ThreadLocalID = int(gl_LocalInvocationID.x);
