Building and running LogPolarGpu on Windows:
    - build project in VisualStudio (project created in VS2015, using OpenCV 2.4.13, tested in x64 Release configuration)
    - run tests using scripts from LogPolarGpu/windows:
        - run all.bat to run all tests
        - use separate .bat files to run particular tests
    - call program directly: 
        x64/Release/LogPolarGpu.exe
    

Building and running on Linux:
    - build using LogPolarGpu/linux/makeLPomp.sh (adjust location of cuda, openMP and OpenCV headers and libs)
    - run tests using scripts from LogPolarGpu/linux:
        - run all.sh to run all tests
        - use separate .sh files to run particular tests
    - call program directly:
        linux/Release/logPolarGpu
        
        
Program usage: 
    LogPolarGpu PATCH_NAME BG_NAME IMAGE_DIR [ANGLE_LIMIT=360] [CONFIDENCE_LEVEL=80] [ROWS_PER_KERNEL=1]
    Example: x64/Release/LogPolarGpu.exe all_souls oxford_003056 all_souls 0 80
    Arguments:
    PATCH_NAME - search image file name (w/o extension) from 'LogPolarTest/IMAGE_DIR/patterns' directory
    BG_NAME - reference image file name (w/o extension) from 'LogPolarTest/IMAGE_DIR/images' directory
    IMAGE_DIR - directory from 'LogPolarTest' directory
    ANGLE_LIMIT - default 360 (full search in angle axis), can be set to any value from range [0..360] to limit computations
    CONFIDENCE_LEVEL - default 80%, can be set to other from range [0..100]. Results images for matches lower than this limit are not saved in result/ directory
    ROWS_PER_KERNEL = default 1, can be changed to higher value to optimize execution time on more powerfull GPUs while operating on small images ang many GPUs
    
Project structure:

- LogPolarGpu
    - linux
        - Release
            LogPolarGpu
    - x64
        - Release
            LogPolarGpu.exe
    - LogPolarGpu
        - linux
            makeLPomp.sh
            all.sh
            test_1.sh
            ...
        - windows
            all.bat
            test_1.bat
            ...
        - LogPolarTest
            - test_1
                - images
                    bg1.jpg
                    bg2.jpg
                    ...
                - patterns
                    patch1.jpg
                    patch2.jpg
                    ...
                - results
            - test_1
                - images
                - patterns
                - results
            - test_3
            ...
        
