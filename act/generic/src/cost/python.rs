use std::path::PathBuf;
use std::sync::OnceLock;

use pyo3::types::{PyAnyMethods, PyModule};
use pyo3::{Py, PyAny, PyErr, Python};

// Cache the Python `model.cost` callable so we only import it once.
static COST_FUNC: OnceLock<Py<PyAny>> = OnceLock::new();

fn init_cost_func() -> &'static Py<PyAny> {
    COST_FUNC.get_or_init(|| {
        let this_file_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(file!());
        let pwd = this_file_path
            .parent()
            .expect("Failed to get rewrites/ directory")
            .join("../../python/cost");
        std::env::set_var("PYTHONPATH", &pwd);
        let load_cost_fn: Result<Py<PyAny>, PyErr> = Python::attach(|py| {
            let module = PyModule::import(py, "model")?;
            let func = module.getattr("cost")?;
            Ok(func.into())
        });
        load_cost_fn.expect("Failed to load cost function")
    })
}

pub fn python_bridge(asm_file_path: &PathBuf) -> i32 {
    let py_func = init_cost_func();

    Python::attach(|py| {
        let arg = asm_file_path.to_str().unwrap();
        py_func
            .call1(py, (arg,))
            .expect("Failed to call cost function")
            .extract::<i32>(py)
            .expect("Failed to extract cost as i32")
    })
}
