// src/combined_math.rs

use pyo3::prelude::*;
use pyo3::types::PyString;
use pyo3::exceptions::{PyTypeError, PyValueError};
use std::str::FromStr;

use ethnum::{U256 as EthU256, I256};
use primitive_types::{U256 as PtU256, U512};
use lazy_static::lazy_static;

// Tick-Bounds aus Uniswap V3
const MIN_TICK: i32 = -887272;
const MAX_TICK: i32 =  887272;

lazy_static! {
    static ref MIN_SQRT_RATIO: EthU256 =
        EthU256::from_str("4295128739").unwrap();
    static ref MAX_SQRT_RATIO: EthU256 =
        EthU256::from_str("1461446703485210103287273052203988822378723970342").unwrap();

    static ref C1: EthU256 =
        EthU256::from_str("3402992956809132418596140100660247210").unwrap();
    static ref C2: EthU256 =
        EthU256::from_str("291339464771989622907027621153398088495").unwrap();
    static ref SCALAR: EthU256 =
        EthU256::from_str("255738958999603826347141").unwrap();
}

lazy_static! {
    static ref MAX_UINT128: EthU256 = EthU256::MAX >> 128;
}

/// Ethnum → Primitive-Types
fn eth_to_pt(x: EthU256) -> PtU256 {
    let bytes = x.to_le_bytes();
    PtU256::from_little_endian(&bytes)
}

/// Primitive-Types → Ethnum
fn pt_to_eth(x: PtU256) -> EthU256 {
    let bytes: [u8; 32] = x.to_little_endian();
    EthU256::from_le_bytes(bytes)
}

/// Berechnet sqrt-Ratio Q128.96 für einen Tick (Uniswap V3)
fn compute_sqrt_ratio_at_tick_inner(tick: i32) -> EthU256 {
    let abs_tick = (tick as i64).abs() as u32;
    
    // 1) Basis-Ratio
    let mut ratio: EthU256 = if abs_tick & 1 != 0 {
        EthU256::from_str("340265354078544963557816517032075149313").unwrap()
    } else {
        *MAX_UINT128
    };
    
    // 2) Faktor-Loop (Q128.128 multiplizieren)
    for &(mask, mul_str) in &[
        (2,  "340248342086729790484326174814286782778"),
        (4,  "340214320654664324051920982716015181260"),
        (8,  "340146287995602323631171512101879684304"),
        (16, "340010263488231146823593991679159461444"),
        (32, "339738377640345403697157401104375502016"),
        (64, "339195258003219555707034227454543997025"),
        (128,"338111622100601834656805679988414885971"),
        (256,"335954724994790223023589805789778977700"),
        (512,"331682121138379247127172139078559817300"),
        (1024,"323299236684853023288211250268160618739"),
        (2048,"307163716377032989948697243942600083929"),
        (4096,"277268403626896220162999269216087595045"),
        (8192,"225923453940442621947126027127485391333"),
        (16384,"149997214084966997727330242082538205943"),
        (32768,"66119101136024775622716233608466517926"),
        (65536,"12847376061809297530290974190478138313"),
        (131072,"485053260817066172746253684029974020"),
        (262144,"691415978906521570653435304214168"),
        (524288,"1404880482679654955896180642"),
    ] {
       if abs_tick & mask != 0 {
            let m = EthU256::from_str(mul_str).unwrap();
            // in PT-Typ umwandeln
            let pr: PtU256 = eth_to_pt(ratio);
            let pm: PtU256 = eth_to_pt(m);
            // 512×512→512, >>128
            let wide: U512 = U512::from(pr) * U512::from(pm);
            let shifted = wide >> 128;
            let pt_val = PtU256::try_from(shifted)
                .expect("High 256 Bits sollten 0 sein");
            ratio = pt_to_eth(pt_val);
        }
    }
    
    // 3) invertiere für positive Ticks
    if tick > 0 {
        ratio = EthU256::MAX / ratio;
    }
    
   // Q128.128 → Q128.96 (div 2^32, round up)
    let shift = EthU256::ONE << 32;
    let rem: EthU256 = ratio & (shift - EthU256::ONE);
    (ratio >> 32) + if rem == EthU256::ZERO { EthU256::ZERO } else { EthU256::ONE }
}

#[pyfunction]
/// Python-API: sqrt-Preis Q128.96 für `tick`
pub fn get_sqrt_ratio_at_tick(py: Python, tick: i32) -> PyResult<Py<PyString>> {
    if tick < MIN_TICK || tick > MAX_TICK {
        return Err(PyErr::new::<PyValueError, _>(
            format!("tick {} out of range [{}, {}]", tick, MIN_TICK, MAX_TICK)
        ));
    }
    let s = compute_sqrt_ratio_at_tick_inner(tick).to_string();
    Ok(PyString::new(py, &s).into())
}

#[pyfunction]
/// Python-API: Tick für gegebenen sqrt-Preis Q128.96 - SCHRITT 2: LOG2 SKALIERUNG + NORMALISIERUNG
pub fn get_tick_at_sqrt_ratio(py: Python, price_obj: PyObject) -> PyResult<i32> {
    // 1) sqrt_price_x96 extrahieren
    let sqrt_price_x96: EthU256 = match price_obj.extract::<u128>(py) {
        Ok(v) => EthU256::from(v),
        Err(_) => {
            let s: String = price_obj
                .extract::<String>(py)
                .map_err(|_| PyErr::new::<PyTypeError, _>(
                    "sqrt_price_x96 must be int (fits in u128) or decimal string"
                ))?;
            EthU256::from_str(&s)
                .map_err(|e| PyErr::new::<PyValueError, _>(format!("invalid: {}", e)))?
        }
    };
    
    println!("[STEP1] sqrt_price_x96 = {}", sqrt_price_x96);

    // 2) Range-Check
    if sqrt_price_x96 < *MIN_SQRT_RATIO || sqrt_price_x96 >= *MAX_SQRT_RATIO {
        return Err(PyErr::new::<PyValueError, _>("sqrt_price_x96 out of range"));
    }

    // 3) in Q128.128-Domain
    let ratio: EthU256 = sqrt_price_x96 << 32;
    println!("[STEP2] ratio Q128.128 = {}", ratio);

    // 4) MSB-Suche - VEREINFACHTE VERSION ERST
    let mut msb = 0u32;
    let mut temp = ratio;
    
    // Einfache MSB-Suche (bit-by-bit)
    while temp > EthU256::ONE {
        temp >>= 1;
        msb += 1;
    }
    
    println!("[STEP3] msb = {}", msb);
    
    // 5) log2 mit korrekter Skalierung (wie in Python: log2 = log2_initial << 64)
    let log2_initial = (msb as i32) - 128;
    let mut log2: I256 = I256::from(log2_initial) << 64;
    println!("[STEP4] log2_initial = {}, log2_scaled = {}", log2_initial, log2);
    
    // 6) Normalisierung von r (genau wie in Python)
    let mut r: EthU256;
    let most_significant_bit_for_max_int128 = 128;
    
    if msb >= most_significant_bit_for_max_int128 {
        r = ratio >> (msb - most_significant_bit_for_max_int128);
        println!("[STEP5] normalize >> {} = {}", msb - most_significant_bit_for_max_int128, r);
    } else {
        r = ratio << (most_significant_bit_for_max_int128 - msb);
        println!("[STEP5] normalize << {} = {}", most_significant_bit_for_max_int128 - msb, r);
    }
    
    // 7) ERSTE ITERATION DES FRACTIONAL LOOPS ZUM TESTEN
    // Bit 63 testen
    let r_squared = r * r;
    let z = r_squared >> 127;
    let f = if z >> 128 > EthU256::ZERO { 1 } else { 0 };
    
    println!("[STEP6] First frac iteration:");
    println!("         r = {}", r);
    println!("         r² = {}", r_squared);
    println!("         z = r² >> 127 = {}", z);
    println!("         z >> 128 = {}", z >> 128);
    println!("         f = {}", f);
    
    if f == 1 {
        log2 = log2 | (I256::ONE << 63);
        println!("         BIT 63 SET!");
    }
    
    r = z >> f;
    println!("         new r = z >> {} = {}", f, r);
    println!("         new log2 = {}", log2);
    
    // TEMPORÄRER RÜCKGABEWERT ZUM DEBUGGEN
    Ok(log2.as_i32())
}