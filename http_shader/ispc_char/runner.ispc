const varying int int_cast(const varying unsigned int32 v) { return (int32)v; }

#include "httpd.ispc"


export void runner_dispatch(uniform int work_group_ID[3], uniform int work_groups[3], 
	uniform struct outputBuffer& v_656,  uniform struct inputBuffer& v_613,  uniform struct heapBuffer& _901
)
{
    uniform int3 gl_NumWorkGroups = int3(work_groups[0], work_groups[1], work_groups[2]);
    uniform int3 gl_WorkGroupID = int3(work_group_ID[0], work_group_ID[1], work_group_ID[2]);
    
    // Vectorise the workgroup
    foreach_tiled(lx = 0 ... gl_WorkGroupSize.x)
    {
        varying int3 gl_LocalInvocationID = int3(lx, 0, 0);
        varying int3 gl_GlobalInvocationID = gl_WorkGroupID * gl_WorkGroupSize + gl_LocalInvocationID;
        varying int gl_LocalInvocationIndex = gl_LocalInvocationID.z * gl_WorkGroupSize.x * gl_WorkGroupSize.y + gl_LocalInvocationID.y * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
        
        httpd_ispc_main( 
	        v_613,
	        v_656,
	        gl_NumWorkGroups,
	        gl_GlobalInvocationID,
	        _901
        );
    }
}

task
void runner_task(uniform int work_groups[3],
	uniform struct outputBuffer& v_656,  uniform struct inputBuffer& v_613,  uniform struct heapBuffer& _901
)
{
    uniform int workGroupID[3];
    workGroupID[0] = taskIndex0;
    workGroupID[1] = taskIndex1;
    workGroupID[2] = taskIndex2;
    runner_dispatch(workGroupID, work_groups,
        v_656, v_613, _901
    );
}

export void runner_main(uniform int work_groups[3],
	uniform struct inputBuffer& v_613,  
	uniform struct outputBuffer& v_656,  
	uniform struct heapBuffer& _901
)
{
    launch[work_groups[0], work_groups[1], work_groups[2]]
    runner_task(work_groups, 
        v_656, v_613, _901
    );
}

