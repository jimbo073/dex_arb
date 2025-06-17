// src/combined_math.rs - NUR SDK TICK-FUNKTIONEN

use pyo3::prelude::*;
use pyo3::types::PyString;
use pyo3::exceptions::{PyTypeError, PyValueError};

// Import der validierten Uniswap V3 Math SDK
use uniswap_v3_math::tick_math;
use ruint::aliases::U256;  // Für SDK-kompatible U256

// Tick-Bounds (gleich wie vorher)
const MIN_TICK: i32 = -887272;
const MAX_TICK: i32 = 887272;

#[pyfunction]
/// Python-API: sqrt-Preis Q128.96 für `tick` - NUTZT UNISWAP V3 MATH SDK
pub fn get_sqrt_ratio_at_tick(py: Python, tick: i32) -> PyResult<Py<PyString>> {
    // Range-Check
    if tick < MIN_TICK || tick > MAX_TICK {
        return Err(PyErr::new::<PyValueError, _>(
            format!("tick {} out of range [{}, {}]", tick, MIN_TICK, MAX_TICK)
        ));
    }
    
    // SDK-Aufruf
    match tick_math::get_sqrt_ratio_at_tick(tick) {
        Ok(sqrt_ratio) => {
            let s = sqrt_ratio.to_string();
            println!("[SDK] get_sqrt_ratio_at_tick({}) = {}", tick, s);
            Ok(PyString::new(py, &s).into())
        }
        Err(e) => Err(PyErr::new::<PyValueError, _>(
            format!("SDK error for tick {}: {:?}", tick, e)
        ))
    }
}

#[pyfunction]
/// Python-API: Tick für gegebenen sqrt-Preis Q128.96 - NUTZT UNISWAP V3 MATH SDK
pub fn get_tick_at_sqrt_ratio(py: Python, price_obj: PyObject) -> PyResult<i32> {
    // sqrt_price_x96 extrahieren (gleiche Logik wie vorher)
    let sqrt_price_x96: u128 = match price_obj.extract::<u128>(py) {
        Ok(v) => v,
        Err(_) => {
            let s: String = price_obj
                .extract::<String>(py)
                .map_err(|_| PyErr::new::<PyTypeError, _>(
                    "sqrt_price_x96 must be int (fits in u128) or decimal string"
                ))?;
            s.parse::<u128>()
                .map_err(|e| PyErr::new::<PyValueError, _>(format!("invalid number: {}", e)))?
        }
    };
    
    println!("[SDK] get_tick_at_sqrt_ratio input: {}", sqrt_price_x96);
    
    // Konvertiere u128 zu U256 (SDK-kompatibel)
    let sqrt_price_u256 = U256::from(sqrt_price_x96);
    
    // SDK-Aufruf
    match tick_math::get_tick_at_sqrt_ratio(sqrt_price_u256) {
        Ok(tick) => {
            println!("[SDK] get_tick_at_sqrt_ratio result: {}", tick);
            Ok(tick)
        }
        Err(e) => Err(PyErr::new::<PyValueError, _>(
            format!("SDK error for sqrt_price_x96 {}: {:?}", sqrt_price_x96, e)
        ))
    }
}

#[pyfunction]
/// Optimierte V2-Berechnung - DIREKT kompatibel mit deiner xi_for_v2_vec Funktion
/// 
/// Input: gleiche Datenstruktur wie deine bestehende Python-Version
/// Output: numpy-Array-kompatible Vec<f64>
pub fn v2_xi(
    _py: Python,
    lambda: f64,
    v2_rx: Vec<f64>,      // deine self._v2_Rx
    v2_ry: Vec<f64>,      // deine self._v2_Ry  
    v2_fee: Vec<f64>,     // deine self._v2_fee
    max_ins: Vec<f64>     // deine Mi_v2 (max_ins für V2-Pools)
) -> PyResult<Vec<f64>> {
    
    let n = v2_rx.len();
    if v2_ry.len() != n || v2_fee.len() != n || max_ins.len() != n {
        return Err(PyErr::new::<PyValueError, _>("All arrays must have same length"));
    }
    
    let mut results = Vec::with_capacity(n);
    
    // EXAKT deine Python-Logik, nur in Rust für Speed
    for i in 0..n {
        let rx = v2_rx[i];
        let ry = v2_ry[i]; 
        let fee = v2_fee[i];
        let mi = max_ins[i];
        
        // Deine Original-Formeln:
        let f0_act = ry * fee / rx;
        let fmax_act = (ry * fee * rx) / ((rx + mi * fee).powi(2));
        
        let xi = if lambda < fmax_act {
            mi  // mask_full
        } else if lambda > f0_act {
            0.0 // mask_zero  
        } else {
            // mask_mid - deine inverse Formel
            let sqrt_term = (ry * fee * rx / lambda).sqrt();
            let xi_raw = (sqrt_term - rx) / fee;
            xi_raw.max(0.0).min(mi)  // np.clip
        };
        
        results.push(xi);
    }
    
    Ok(results)
}

// ========================================================================
// V3 OPTIMIZATION - NEUE KOMPLEXE IMPLEMENTIERUNG
// ========================================================================

#[pyfunction]
/// Ultra-optimierte V3-Berechnung mit SDK Tick-Quantisierung
/// 
/// Mathematik: Concentrated Liquidity + Tick-Quantisierung
/// 1. sqrtP* = √(λ / fee_factor)                    // Ziel-Preis aus Lambda
/// 2. x_cont = L × (1/√P* - 1/√P0)                  // Continuous Input
/// 3. tick_raw = SDK.get_tick_at_sqrt_ratio(√P*)    // Preis → Tick
/// 4. tick_q = (tick_raw ÷ spacing) × spacing       // Quantisierung
/// 5. sqrtP_final = SDK.get_sqrt_ratio_at_tick()    // Tick → Preis
/// 6. x_final = L × (1/√P_final - 1/√P0) ÷ fee     // Final Input
///
/// Performance: SDK-Integration + Vektorisierung für mehrere V3-Pools
pub fn calculate_v3_xi_rust(
    _py: Python,
    lambda: f64,
    liquidities: Vec<f64>,        // [L1, L2, L3, ...] Liquidity pro Pool
    sqrt_prices_x96: Vec<u128>,   // [sqrtP0_1, sqrtP0_2, ...] Aktuelle Preise
    fee_rates: Vec<f64>,          // [0.0001, 0.0005, 0.003, ...] Fee-Raten
    tick_spacings: Vec<i32>,      // [1, 10, 60, ...] Tick-Spacings pro Pool
    max_inputs: Vec<f64>          // [Mi1, Mi2, Mi3, ...] Max-Inputs pro Pool
) -> PyResult<Vec<f64>> {
    
    let n = liquidities.len();
    if sqrt_prices_x96.len() != n || fee_rates.len() != n || 
       tick_spacings.len() != n || max_inputs.len() != n {
        return Err(PyErr::new::<PyValueError, _>("All input vectors must have same length"));
    }
    
    if n == 0 {
        return Ok(vec![]);
    }
    
    let mut results = Vec::with_capacity(n);
    
    for i in 0..n {
        let liquidity = liquidities[i];
        let sqrt_price_x96 = sqrt_prices_x96[i];
        let fee_rate = fee_rates[i];
        let tick_spacing = tick_spacings[i];
        let max_input = max_inputs[i];
        
        // Validierung
        if liquidity <= 0.0 || sqrt_price_x96 == 0 || fee_rate < 0.0 || max_input <= 0.0 {
            results.push(0.0);
            continue;
        }
        
        // 1) Fee-Factor und aktuelle sqrt-Price
        let fee_factor = 1.0 - fee_rate;
        let sqrt_p0 = (sqrt_price_x96 as f64) / (2.0_f64.powi(96));
        
        // 2) Ziel sqrt-Preis aus Lambda
        let sqrt_p_star = (lambda / fee_factor).sqrt();
        
        if sqrt_p_star <= 0.0 {
            results.push(0.0);
            continue;
        }
        
        // 3) Continuous Input (erste Approximation)
        let x_cont = liquidity * (1.0/sqrt_p_star - 1.0/sqrt_p0);
        let x_cont_clamped = x_cont.max(0.0).min(max_input);
        
        // 4) SDK Tick-Quantisierung (der kritische Teil!)
        let sqrt_p_star_x96 = (sqrt_p_star * 2.0_f64.powi(96)) as u128;
        
        match tick_math::get_tick_at_sqrt_ratio(U256::from(sqrt_p_star_x96)) {
            Ok(raw_tick) => {
                // Quantisiere auf Tick-Spacing
                let quantized_tick = if tick_spacing > 0 {
                    (raw_tick / tick_spacing) * tick_spacing
                } else {
                    raw_tick
                };
                
                // Exaktes sqrt-Price für quantisierten Tick
                match tick_math::get_sqrt_ratio_at_tick(quantized_tick) {
                    Ok(sqrt_ratio_final_u256) => {
                        let sqrt_ratio_final_str = sqrt_ratio_final_u256.to_string();
                        
                        // Parse zurück zu f64 (mit Präzisionsverlust, aber konsistent)
                        if let Ok(sqrt_ratio_final_u128) = sqrt_ratio_final_str.parse::<u128>() {
                            let sqrt_p_final = (sqrt_ratio_final_u128 as f64) / 2.0_f64.powi(96);
                            
                            // Finaler Input basierend auf quantisiertem Preis
                            let x_final = liquidity * (1.0/sqrt_p_final - 1.0/sqrt_p0) / fee_factor;
                            let x_final_clamped = x_final.max(0.0).min(max_input);
                            
                            results.push(x_final_clamped);
                            
                            // Debug für ersten Pool
                            if i == 0 {
                                println!("[V3_RUST] Pool 0: L={:.3e}, sqrtP0={:.3e}, fee={:.6}", 
                                         liquidity, sqrt_p0, fee_rate);
                                println!("[V3_RUST] Pool 0: sqrtP*={:.3e}, raw_tick={}, q_tick={}", 
                                         sqrt_p_star, raw_tick, quantized_tick);
                                println!("[V3_RUST] Pool 0: x_cont={:.3e}, x_final={:.3e}", 
                                         x_cont_clamped, x_final_clamped);
                            }
                        } else {
                            // Fallback auf continuous bei Parse-Problemen
                            results.push(x_cont_clamped);
                        }
                    }
                    Err(_) => {
                        // Fallback auf continuous bei SDK-Problemen
                        results.push(x_cont_clamped);
                    }
                }
            }
            Err(_) => {
                // Fallback auf continuous bei SDK-Problemen
                results.push(x_cont_clamped);
            }
        }
    }
    
    let total_xi: f64 = results.iter().sum();
    println!("[V3_RUST] Total xi_v3 = {:.6e}", total_xi);
    
    Ok(results)
}

// ========================================================================
// G_TOTAL MEGA-OPTIMIZATION - KOMBINIERT ALLES!
// ========================================================================

#[pyfunction]
/// ULTRA-PERFORMANCE: Kombinierte g_total Berechnung für Brent's Optimierung
/// 
/// Diese Funktion macht ALLES in einem Rust-Aufruf:
/// 1. V2-Pool Berechnungen (vektorisiert)  
/// 2. V3-Pool Berechnungen (mit SDK Tick-Quantisierung)
/// 3. Summierung und g_total Ergebnis
/// 
/// Eliminiert hunderte Python↔Rust Transfers während Brent's Root-Finding!
pub fn calculate_g_total_rust(
    _py: Python,
    lambda: f64,
    total_amount_in: f64,
    
    // V2 Pool Daten (Arrays für Batch-Verarbeitung)
    v2_rx: Vec<f64>,
    v2_ry: Vec<f64>, 
    v2_fee: Vec<f64>,
    v2_max_ins: Vec<f64>,
    
    // V3 Pool Daten (Arrays für Batch-Verarbeitung)
    v3_liquidities: Vec<f64>,
    v3_sqrt_prices_x96: Vec<u128>,
    v3_fee_rates: Vec<f64>,
    v3_tick_spacings: Vec<i32>,
    v3_max_inputs: Vec<f64>
) -> PyResult<f64> {
    
    // Validierung
    let n_v2 = v2_rx.len();
    let n_v3 = v3_liquidities.len();
    
    if n_v2 > 0 && (v2_ry.len() != n_v2 || v2_fee.len() != n_v2 || v2_max_ins.len() != n_v2) {
        return Err(PyErr::new::<PyValueError, _>("V2 arrays must have same length"));
    }
    
    if n_v3 > 0 && (v3_sqrt_prices_x96.len() != n_v3 || v3_fee_rates.len() != n_v3 || 
                    v3_tick_spacings.len() != n_v3 || v3_max_inputs.len() != n_v3) {
        return Err(PyErr::new::<PyValueError, _>("V3 arrays must have same length"));
    }
    
    // V2 Berechnung (vektorisiert)
    let mut sum_v2 = 0.0;
    for i in 0..n_v2 {
        let rx = v2_rx[i];
        let ry = v2_ry[i]; 
        let fee = v2_fee[i];
        let max_in = v2_max_ins[i];
        
        if rx <= 0.0 || ry <= 0.0 || fee <= 0.0 || max_in <= 0.0 {
            continue;
        }
        
        // Exakt deine V2-Logik
        let f0_act = ry * fee / rx;
        let fmax_act = (ry * fee * rx) / ((rx + max_in * fee).powi(2));
        
        let xi = if lambda < fmax_act {
            max_in
        } else if lambda > f0_act {
            0.0
        } else {
            let sqrt_term = (ry * fee * rx / lambda).sqrt();
            let xi_raw = (sqrt_term - rx) / fee;
            xi_raw.max(0.0).min(max_in)
        };
        
        sum_v2 += xi;
    }
    
    // V3 Berechnung (mit SDK-Integration)
    let mut sum_v3 = 0.0;
    for i in 0..n_v3 {
        let liquidity = v3_liquidities[i];
        let sqrt_price_x96 = v3_sqrt_prices_x96[i];
        let fee_rate = v3_fee_rates[i];
        let tick_spacing = v3_tick_spacings[i];
        let max_input = v3_max_inputs[i];
        
        if liquidity <= 0.0 || sqrt_price_x96 == 0 || fee_rate < 0.0 || max_input <= 0.0 {
            continue;
        }
        
        // Exakt deine V3-Logik (aber optimiert)
        let fee_factor = 1.0 - fee_rate;
        let sqrt_p0 = (sqrt_price_x96 as f64) / (2.0_f64.powi(96));
        let sqrt_p_star = (lambda / fee_factor).sqrt();
        
        if sqrt_p_star <= 0.0 {
            continue;
        }
        
        // Continuous approximation als Fallback
        let x_cont = liquidity * (1.0/sqrt_p_star - 1.0/sqrt_p0);
        let x_cont_clamped = x_cont.max(0.0).min(max_input);
        
        // SDK Tick-Quantisierung (mit Fallback bei Fehlern) 
        let sqrt_p_star_x96 = (sqrt_p_star * 2.0_f64.powi(96)) as u128;
        
        let xi = match tick_math::get_tick_at_sqrt_ratio(U256::from(sqrt_p_star_x96)) {
            Ok(raw_tick) => {
                let quantized_tick = if tick_spacing > 0 {
                    (raw_tick / tick_spacing) * tick_spacing
                } else {
                    raw_tick
                };
                
                match tick_math::get_sqrt_ratio_at_tick(quantized_tick) {
                    Ok(sqrt_ratio_final_u256) => {
                        let sqrt_ratio_final_str = sqrt_ratio_final_u256.to_string();
                        
                        if let Ok(sqrt_ratio_final_u128) = sqrt_ratio_final_str.parse::<u128>() {
                            let sqrt_p_final = (sqrt_ratio_final_u128 as f64) / 2.0_f64.powi(96);
                            let x_final = liquidity * (1.0/sqrt_p_final - 1.0/sqrt_p0) / fee_factor;
                            x_final.max(0.0).min(max_input)
                        } else {
                            x_cont_clamped  // Fallback
                        }
                    }
                    Err(_) => x_cont_clamped  // Fallback
                }
            }
            Err(_) => x_cont_clamped  // Fallback
        };
        
        sum_v3 += xi;
    }
    
    // G_TOTAL Berechnung (wie in deiner Python-Version)
    let g_total_result = sum_v2 + sum_v3 - total_amount_in;
    
    // Minimales Debug (um Performance nicht zu beeinträchtigen)
    if lambda.abs() < 1e-12 || lambda > 1e12 {  // Nur bei extremen Lambda-Werten
        println!("[G_TOTAL_RUST] λ={:.3e}, V2={:.3e}, V3={:.3e}, result={:.3e}", 
                 lambda, sum_v2, sum_v3, g_total_result);
    }
    
    Ok(g_total_result)
}