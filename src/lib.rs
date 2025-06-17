extern crate lazy_static;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::types::{PyModule, PyModuleMethods}; // PyModuleMethods für add_wrapped
pub mod combined_math;

#[pymodule(name = "combined_math")]
fn init_combined_math(_py: Python<'_>, m: pyo3::Bound<'_, PyModule>) -> PyResult<()> {
    // `m` ist jetzt Bound<'py, PyModule>, das derefs zu &PyModule
    m.add_wrapped(wrap_pyfunction!(combined_math::get_sqrt_ratio_at_tick))?;
    m.add_wrapped(wrap_pyfunction!(combined_math::get_tick_at_sqrt_ratio))?;
    // Minimal invasive V2-Optimierung 
    m.add_wrapped(wrap_pyfunction!(combined_math::v2_xi))?;
    m.add_wrapped(wrap_pyfunction!(combined_math::calculate_v3_xi_rust))?;
    m.add_wrapped(wrap_pyfunction!(combined_math::calculate_g_total_rust))?;
    Ok(())
}
