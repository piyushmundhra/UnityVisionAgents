src_dir="./build"
dest_dir="/Users/piyushmundhra/neighborhood/Assets/Plugins/macOS/"
trash_dir=~/.Trash

echo -e "\033[0;34mQuitting Unity...\033[0m"
osascript -e 'quit app "Unity"'

echo -e "\033[0;34mMoving all existing libs in $dest_dir to Trash\033[0m"
mv "$dest_dir"/* "$trash_dir"

echo -e "\033[0;34mCopying files\033[0m"
cp "$src_dir"/*.dylib "$dest_dir"

echo -e "\033[1;32mStarting Unity Hub\033[0m"
open -a "Unity Hub"