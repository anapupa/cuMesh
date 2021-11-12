add_executable(TestCudaConvertHedge TestCudaConvertHedge.cpp)
target_link_libraries (TestCudaConvertHedge
        cuMesh
        cuMesh_Impl
        )
