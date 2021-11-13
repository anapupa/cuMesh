add_executable(TestCudaConvertHedge TestCudaConvertHedge.cpp)
target_link_libraries (TestCudaConvertHedge
        cuMesh
        cuMesh_Impl
        polyscope
        )


add_executable(TestPQGF TestCudaPGGF.cpp)
target_link_libraries (TestPQGF
        cuMesh
        cuMesh_Impl
        polyscope
        )
