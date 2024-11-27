# Script to Install Tilt

if tilt > /dev/null 2>&1; then
    echo "Tilt: Its already installed....."
    exit 0
fi

curl -fsSL https://raw.githubusercontent.com/tilt-dev/tilt/master/scripts/install.sh | bash

if tilt > /dev/null 2>&1; then
    echo "Tilt: Installation Successful"
else
    echo "ERROR: Tilt installation was not successfull"
    echo "Check if tilt installation directory is in PATH"
    exit 1
fi


