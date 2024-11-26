#!/bin/bash

##### Functions ##############################

show_help() {
    echo "IAC: Script 1"
    echo "This script installs K3s and required supporting binaries (such as kubectl)"
    echo
    echo "Usage: $0 [options]"
    echo 
    echo "Options (REQUIRED in ORDER)"
    echo "  1. <os>,   [windows, darwin, linux]"
    echo "  2. <arch>, [amd64, arm64]"
    echo
    echo "Example:"
    echo "./1-install-k3s.sh darwin arm64"
    echo
    echo "NOTES:"
    echo "1. Operating system option 'darwin' is for MacOS"
}

check_os() {
    local OS="$1"
    
    case "$OS" in
        # Match Windows
        [Ww][Ii][Nn][Dd][Oo][Ww][Ss])
            echo "OS Selection: Windows-NT"
            ;;

        # Match MacOS
        [Dd][Aa][Rr][Ww][Ii][Nn])
            echo "OS Selection: Darwin(MacOS)"
            ;;

        # Match Linux
        [Ll][Ii][Nn][Uu][Xx])
            echo "OS Selection: Linux"
            ;;

        # Unsupported Case
        *)
            echo "ERROR: OS Selection Unsupported"
            exit 1
            ;;
    esac
}


check_arch() {
    local ARCH="$1"

    case "$ARCH" in
        # Match x86
        [Aa][Mm][Dd][6][4])
            echo "Arch Detection: x86"
            ;;

        # Match ARM
        [Aa][Rr][Mm][6][4])
            echo "Arch Detection: ARM64"
            ;;

        # Unsupported Cases
        *)
            echo "ERROR: CPU Architecture Selection Unsupported"
            return 1
            ;;
    esac
}

######### MAIN SCRIPT ################################################
if [[ "$#" -eq 0 ]]; then
    
    # No args are passed
    show_help
    exit 0
fi

OS=$1
ARCH=$2

# Arg Validation
check_os "$OS"
check_arch "$ARCH"

# Install Kubectl
if [ ! -z $1 ]; then
    curl -f -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/$OS/$ARCH/kubectl"
fi

# Move Kubectl to local bin
DIRECTORY="$HOME/.local/bin"
if [ ! -d $DIRECTORY ]; then
    mkdir -p $DIRECTORY
fi

if [ -f $DIRECTORY/kubectl ]; then
    rm $DIRECTORY/kubectl
fi

mv ./kubectl $DIRECTORY/
chmod +x $DIRECTORY/kubectl

# Reinitialize Script Environment
source $HOME/.bashrc

# Check to See if Installation Successful
if kubectl > /dev/null 2>&1; then
    echo "KUBECTL: INSTALLATION SUCCESSFULL"
    echo
else
    echo "ERROR: INSTALLATION WAS NOT SUCCESFULL"
    echo "Check if $DIRECTORY is in PATH"
fi  


