init:
    git submodule update --init --recursive

build: init
    cargo build

test: init
    cargo test
