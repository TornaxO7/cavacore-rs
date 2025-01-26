use std::{fs::File, io::Write, path::PathBuf};

fn main() {
    cc::Build::new()
        .file("./cava/cavacore.c")
        .static_flag(true)
        .compile("cavacore");

    let cava_dir = PathBuf::from("./cava")
        .canonicalize()
        .expect("Canonicalize path");

    let header_path = cava_dir.join("cavacore.h");
    let header_path_str = header_path.to_str().expect("Get path to header as string");

    println!(
        "cargo:rustc-link-search={}",
        std::env::var("OUT_DIR").unwrap()
    );
    println!("cargo:rustc-link-lib=cavacore");
    println!("cargo:rustc-link-lib=fftw3");

    let bindings = bindgen::Builder::default()
        .header(header_path_str)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Generate bindings");

    let out_path = PathBuf::from("./src").join("bindings.rs");
    let mut bindings_file = Box::new(File::create(out_path).unwrap());

    bindings_file.write_all(b"#![allow(warnings)]\n").unwrap();

    bindings
        .write(bindings_file)
        .expect("Couldn't write bindings!");
}
