libomp_path="/opt/homebrew/opt/libomp/lib"
CYAN='\033[0;36m'
BLUE_BOLD='\033[1;34m'
NC='\033[0m' 

echo -e "${CYAN}Starting build script${NC}"

if [ -d "build" ]; then
    rm -rf build
    mkdir build
    echo -e "${CYAN}Cleared build folder${NC}"
else
    mkdir build
    echo -e "${CYAN}Created build folder${NC}"
fi

cd build

echo -e "${CYAN}Building...${NC}"
export MACOSX_DEPLOYMENT_TARGET=14.0
cmake ..
make


for dylib_file in *.dylib
do  

    echo -e "${CYAN}Configuring $dylib_file${NC}"
    install_name_tool -add_rpath $libomp_path $dylib_file

    echo -e "${BLUE_BOLD}Verify that libtorch, onnx, libomp paths are present for $dylib_file:${NC}"
    otool -l $dylib_file | grep -A2 LC_RPATH | sed 's/^/\t/'

done

# after running 
    # brew install libomp
    # brew info libomp
    
# for executable:
    # export DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH

# when importing libagent_lib.dylib to unity: 
    # if "*.dylib can't be opened because macOS can't verify blah blah blah"
        # privacy & security > security > manually allow each one to run. 
        # privacy & security > developer tools > add unity as an app that can run software that does not meet security requirements
            # (might have to navigate in the unity folder to find the actual application file)  