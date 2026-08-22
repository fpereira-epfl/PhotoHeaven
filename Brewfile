# PhotoHeaven Brewfile
#
# This file declares the system-level dependencies that PhotoHeaven needs.
# Python packages are managed separately via pyproject.toml / pip.
#
# How to use this file:
#
#   1. Make sure you have Homebrew installed:
#        https://brew.sh
#
#   2. Run brew bundle from the root of this repository:
#        brew bundle
#
#   3. Homebrew will install everything listed below.
#
#   4. Continue with the Python setup:
#        python -m venv .venv
#        source .venv/bin/activate
#        pip install -e ".[dev]"
#
# After that you can use the CLI:
#        ph --help
#

# MediaInfo is required by pymediainfo to read container metadata from video
# files such as .mov, .mp4, .mts, .m2ts, .avi, and .mkv.
brew "mediainfo"
