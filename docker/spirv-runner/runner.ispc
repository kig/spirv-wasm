#include "program.ispc"

export void runner_workgroup(uniform int numWorkGroups[3], uniform int workGroupID[3], uniform struct inputs& input, uniform struct outputs& output)
{
    uniform int3 gl_NumWorkGroups = int3(numWorkGroups[0], numWorkGroups[1], numWorkGroups[2]);
    uniform int3 gl_WorkGroupID = int3(workGroupID[0], workGroupID[1], workGroupID[2]);

    // Vectorise the workgroup
    foreach(lz = 0 ... gl_WorkGroupSize.z, ly = 0 ... gl_WorkGroupSize.y, lx = 0 ... gl_WorkGroupSize.x)
    {
        varying int3 gl_LocalInvocationID = int3(lx, ly, lz);
        varying int3 gl_GlobalInvocationID = gl_WorkGroupID * gl_WorkGroupSize + gl_LocalInvocationID;
        varying int gl_LocalInvocationIndex = gl_LocalInvocationID.z * gl_WorkGroupSize.x * gl_WorkGroupSize.y + gl_LocalInvocationID.y * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
        
        program_ispc_main(gl_NumWorkGroups, gl_WorkGroupID, gl_LocalInvocationID, gl_GlobalInvocationID, gl_LocalInvocationIndex, input, output);
    }
}

task
void runner_task(uniform int work_groups[3], uniform struct inputs& input, uniform struct outputs& output)
{
    uniform int workGroupID[3];
    workGroupID[0] = taskIndex0;
    workGroupID[1] = taskIndex1;
    workGroupID[2] = taskIndex2;
    runner_workgroup(work_groups, workGroupID, input, output);
}

export void runner_main(uniform int work_groups[3], uniform struct inputs& input, uniform struct outputs& output)
{
    launch[work_groups[0], work_groups[1], work_groups[2]]
    runner_task(work_groups, input, output);
}

export static int stdout = 1;
export static int stderr = 2;