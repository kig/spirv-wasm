int32_t ThreadGroupCount = int(gl_NumWorkGroups.x);
int32_t ThreadLocalCount = int(gl_WorkGroupSize.x);
int32_t ThreadCount = ThreadGroupCount * ThreadLocalCount;
int32_t ThreadId = int(gl_GlobalInvocationID.x);
int32_t ThreadGroupId = int(gl_WorkGroupID.x);
int32_t ThreadLocalId = int(gl_LocalInvocationID.x);
