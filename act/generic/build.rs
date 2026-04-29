use std::{env, fs, path::PathBuf, process::Command};

fn main() {
    println!("cargo:rustc-link-lib=dylib=stdc++");

    println!("cargo:rustc-link-search=native=/opt/ortools/lib");
    println!("cargo:rustc-link-lib=dylib=ortools");
    println!("cargo:rustc-link-lib=dylib=absl_base");
    println!("cargo:rustc-link-lib=dylib=absl_log_internal_check_op");

    // Create cpp/malloc/build, run `cmake ..` inside it, then `make -j4`
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let cpp_build_dir = manifest.join("cpp").join("malloc").join("build");
    fs::create_dir_all(&cpp_build_dir).expect("failed to create cpp/malloc/build");

    let status = Command::new("cmake")
        .arg("..")
        .current_dir(&cpp_build_dir)
        .status()
        .expect("failed to run cmake");
    if !status.success() {
        panic!("cmake failed");
    }

    let status = Command::new("make")
        .arg("-j4")
        .current_dir(&cpp_build_dir)
        .status()
        .expect("failed to run make");
    if !status.success() {
        panic!("make failed");
    }

    println!("cargo:rustc-link-search=native={}", cpp_build_dir.display());
    println!("cargo:rustc-link-lib=static=malloc_static");
}
