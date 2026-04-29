use std::{ffi::CString, path::PathBuf};

use libc::c_char;

use crate::LOG_FILE;

#[link(name = "malloc_static")]
unsafe extern "C" {
    fn act_malloc(
        pii_path: *const c_char,
        metadata_path: *const c_char,
        asm_path: *const c_char,
    ) -> i32;
}

pub fn cpp_bridge(pii_path: &PathBuf, metadata_path: &PathBuf, asm_path: &PathBuf) -> i32 {
    let pii_path = CString::new(pii_path.to_str().unwrap()).unwrap();
    let metadata_path = CString::new(metadata_path.to_str().unwrap()).unwrap();
    let asm_path = CString::new(asm_path.to_str().unwrap()).unwrap();

    // Redirect C stdout to log file if set, otherwise to /dev/null
    let saved_stdout = unsafe { libc::dup(1) };
    {
        let guard = LOG_FILE.lock().unwrap();
        if let Some(ref file) = *guard {
            use std::os::unix::io::AsRawFd;
            unsafe { libc::dup2(file.as_raw_fd(), 1); }
        } else {
            let devnull = CString::new("/dev/null").unwrap();
            let fd = unsafe { libc::open(devnull.as_ptr(), libc::O_WRONLY) };
            unsafe { libc::dup2(fd, 1); libc::close(fd); }
        }
    }

    let result = unsafe {
        act_malloc(pii_path.as_ptr(), metadata_path.as_ptr(), asm_path.as_ptr())
    };

    // Restore stdout
    unsafe {
        libc::fflush(std::ptr::null_mut());
        libc::dup2(saved_stdout, 1);
        libc::close(saved_stdout);
    }

    result
}
