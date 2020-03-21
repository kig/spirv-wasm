#include "program.ispc"

task
void runner_tiled(uniform int3 gl_NumWorkGroups, uniform struct inputs& input, uniform struct outputs& output)
{
    uniform int3 gl_WorkGroupID = int3(taskIndex0, taskIndex1, taskIndex2);

    // Vectorise the workgroup
    foreach(lz = 0 ... gl_WorkGroupSize.z, ly = 0 ... gl_WorkGroupSize.y, lx = 0 ... gl_WorkGroupSize.x)
    {
        varying int3 gl_LocalInvocationID = int3(lx, ly, lz);
        varying int3 gl_GlobalInvocationID = gl_WorkGroupID * gl_WorkGroupSize + gl_LocalInvocationID;
        varying int gl_LocalInvocationIndex = gl_LocalInvocationID.z * gl_WorkGroupSize.x * gl_WorkGroupSize.y + gl_LocalInvocationID.y * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
        
        program_ispc_main(gl_NumWorkGroups, gl_WorkGroupID, gl_LocalInvocationID, gl_GlobalInvocationID, gl_LocalInvocationIndex, input, output);
    }
}

export void runner_main(uniform int work_groups[3], uniform struct inputs& input, uniform struct outputs& output)
{
    uniform int3 gl_NumWorkGroups = int3(work_groups[0], work_groups[1], work_groups[2]);

    launch[work_groups[0], work_groups[1], work_groups[2]]
    runner_tiled(gl_NumWorkGroups, input, output);
}
