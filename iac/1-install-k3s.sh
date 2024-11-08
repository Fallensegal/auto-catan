#!/bin/bash

##### Functions #####

show_help() {
    echo "IAC: Script 1"
    echo "This script installs K3s and required supporting binaries (such as kubectl)"
    echo
    echo "Usage: $0 [options]"
    echo 
    echo "Options (REQUIRED in ORDER)"
    echo "  1. <os>,   [windows, darwin, linux]"
    echo "  2. <arch>, [x86, arm64]"
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
            return 1
            ;;
    esac
}           

######### MAIN SCRIPT #########
if [[ "$#" -eq 0 ]]; then
    
    # No args are passed
    show_help
    exit 0
fi

OS=$1
#ARCH=$2

# Arg Validation
check_os "$OS"

# Install K3s
#if ![ -z $1 ]; then
#    curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/$ARCH/$OS/kubectl"
#fi



