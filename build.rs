use std::{env, path::PathBuf};

fn main() {
    cc::Build::new()
        .file("./cava/cavacore.c")
        .compile("cavacore");

    let libdir_path = PathBuf::from("./cava")
        .canonicalize()
        .expect("Canonicalize path");

    let header_path = libdir_path.join("cavacore.h");
    let header_path_str = header_path.to_str().expect("Path as string");

    let bindings = bindgen::Builder::default()
        .header(header_path_str)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap()).join("bindings.rs");
    bindings
        .write_to_file(out_path)
        .expect("Couldn't write bindings!");
}
